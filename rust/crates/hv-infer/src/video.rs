//! Video frames from the `ffmpeg` command-line tool, decoded on a background thread (like
//! video_io.FrameReader). yuv420p video is read as YUV and converted with swscale's own C
//! converter (hv_core::image::YuvToBgr), so the frames are exactly what cv2.VideoCapture gives.
//! Other pixel formats are converted by ffmpeg itself (not bit-identical to OpenCV). Camera cuts are
//! found on the same thread (video_io.CutDetector), so each frame comes with its cut flag.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread::JoinHandle;

use hv_core::image::{CutDetector, Image, YuvToBgr};

#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    /// avg_frame_rate, as cv2.CAP_PROP_FPS reports it.
    pub fps: f64,
    /// Frames in the container (0 when unknown).
    pub frames: usize,
    pub pix_fmt: String,
}

fn rate(s: &str) -> f64 {
    match s.split_once('/') {
        Some((n, d)) => {
            let (n, d): (f64, f64) = (n.parse().unwrap_or(0.0), d.parse().unwrap_or(0.0));
            if d == 0.0 { 0.0 } else { n / d }
        }
        None => s.parse().unwrap_or(0.0),
    }
}

pub fn probe(path: &Path) -> Result<VideoInfo, String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .arg("stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt")
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .map_err(|e| format!("running ffprobe (is ffmpeg installed?): {e}"))?;
    if !out.status.success() {
        return Err(format!("Could not open {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let get = |k: &str| text.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).unwrap_or("").trim().to_string();
    let mut fps = rate(&get("avg_frame_rate"));
    if fps <= 0.0 {
        fps = rate(&get("r_frame_rate"));
    }
    Ok(VideoInfo {
        width: get("width").parse().map_err(|_| format!("{}: no video stream", path.display()))?,
        height: get("height").parse().unwrap_or(0),
        fps: if fps > 0.0 { fps } else { 30.0 },
        frames: get("nb_frames").parse().unwrap_or(0),
        pix_fmt: get("pix_fmt"),
    })
}

pub struct FrameReader {
    pub info: VideoInfo,
    /// True when frames are bit-identical to OpenCV's decoding.
    pub exact: bool,
    rx: Receiver<Result<(Image, bool), String>>,
    child: Child,
    path: PathBuf,
    thread: Option<JoinHandle<()>>,
    reader_error: Option<String>,
    finished: bool,
    child_waited: bool,
}

impl FrameReader {
    pub fn open(path: &Path, max_frames: usize) -> Result<Self, String> {
        let info = probe(path)?;
        let exact = info.pix_fmt == "yuv420p" && info.width % 2 == 0 && info.height % 2 == 0;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin", "-i"]).arg(path);
        if max_frames > 0 {
            cmd.args(["-frames:v", &max_frames.to_string()]);
        }
        cmd.args(["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt"])
            .arg(if exact { "yuv420p" } else { "bgr24" })
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd.spawn().map_err(|e| format!("running ffmpeg: {e}"))?;
        let mut stdout = child.stdout.take().unwrap();
        let (w, h) = (info.width, info.height);
        let (tx, rx) = sync_channel(8);
        let thread = std::thread::spawn(move || {
            let conv = YuvToBgr::bt601_limited();
            let mut cuts = CutDetector::default();
            let size = if exact { w * h * 3 / 2 } else { w * h * 3 };
            let mut buf = vec![0u8; size];
            loop {
                match read_full(&mut stdout, &mut buf) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => {
                        let _ = tx.send(Err(format!("reading frames: {e}")));
                        break;
                    }
                }
                let img = if exact {
                    let mut img = Image::default();
                    conv.convert(&buf, w, h, &mut img);
                    img
                } else {
                    Image { w, h, data: buf.clone() }
                };
                let cut = cuts.update(&img);
                if tx.send(Ok((img, cut))).is_err() {
                    break; // reader dropped
                }
            }
        });
        Ok(FrameReader {
            info,
            exact,
            rx,
            child,
            path: path.to_path_buf(),
            thread: Some(thread),
            reader_error: None,
            finished: false,
            child_waited: false,
        })
    }

    /// Called after the reader thread closes its channel, which means FFmpeg closed stdout.
    /// Waiting here makes an unsuccessful FFmpeg exit visible instead of treating it as EOF.
    fn finish(&mut self) -> Option<<Self as Iterator>::Item> {
        self.finished = true;

        let reader_panicked = self.thread.take().is_some_and(|thread| thread.join().is_err());
        let status = match self.child.wait() {
            Ok(status) => {
                self.child_waited = true;
                status
            }
            Err(error) => {
                return Some(Err(format!("waiting for ffmpeg while decoding {}: {error}", self.path.display())));
            }
        };

        if let Some(error) = ffmpeg_exit_error(&self.path, &status, self.reader_error.as_deref()) {
            return Some(Err(error));
        }
        if let Some(error) = self.reader_error.take() {
            return Some(Err(format!("frame reader failed while decoding {}: {error}", self.path.display())));
        }
        if reader_panicked {
            return Some(Err(format!("frame reader thread panicked while decoding {}", self.path.display())));
        }
        None
    }
}

fn ffmpeg_exit_error(path: &Path, status: &ExitStatus, reader_error: Option<&str>) -> Option<String> {
    if status.success() {
        return None;
    }

    let mut message = format!("ffmpeg exited with {status} while decoding {}", path.display());
    if let Some(error) = reader_error {
        message.push_str(&format!("; frame reader also reported: {error}"));
    }
    Some(message)
}

/// Fill `buf`; false at a clean end of stream.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..])? {
            0 if got == 0 => return Ok(false),
            0 => return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated frame")),
            n => got += n,
        }
    }
    Ok(true)
}

impl Iterator for FrameReader {
    /// A frame and whether it starts a new camera shot.
    type Item = Result<(Image, bool), String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            match self.rx.recv() {
                Ok(Ok(frame)) => return Some(Ok(frame)),
                // Defer reader errors until FFmpeg's exit status is known, so both can be
                // reported together when FFmpeg failed after emitting a partial stream.
                Ok(Err(error)) => self.reader_error = Some(error),
                Err(_) => return self.finish(),
            }
        }
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        if !self.child_waited {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        // Unblock the thread if it is waiting to send.
        while self.rx.try_recv().is_ok() {}
        if let Some(t) = self.thread.take() {
            drop(std::mem::replace(&mut self.rx, sync_channel(1).1));
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ffmpeg_exit_error;
    use std::path::Path;
    use std::process::ExitStatus;

    #[cfg(unix)]
    fn failed_status() -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(7 << 8)
    }

    #[cfg(windows)]
    fn failed_status() -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(7)
    }

    #[cfg(unix)]
    fn successful_status() -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }

    #[cfg(windows)]
    fn successful_status() -> ExitStatus {
        use std::os::windows::process::ExitStatusExt;
        ExitStatus::from_raw(0)
    }

    #[test]
    fn successful_ffmpeg_exit_is_clean_eof() {
        assert!(ffmpeg_exit_error(Path::new("clip.mp4"), &successful_status(), None).is_none());
    }

    #[test]
    fn nonzero_ffmpeg_exit_is_reported_with_the_video_path() {
        let error = ffmpeg_exit_error(Path::new("clip.mp4"), &failed_status(), None).unwrap();
        assert!(error.contains("ffmpeg exited with"));
        assert!(error.contains("clip.mp4"));
    }

    #[test]
    fn ffmpeg_exit_error_includes_a_reader_error_when_available() {
        let error = ffmpeg_exit_error(Path::new("clip.mp4"), &failed_status(), Some("truncated frame")).unwrap();
        assert!(error.contains("truncated frame"));
    }
}
