#!/usr/bin/env bash
# Run detection and homography for one clip, honoring --output in both stages.
set -euo pipefail
if (( $# == 0 )); then echo 'Usage: run_clip.sh VIDEO [--output DIR] [model options]' >&2; exit 2; fi
video=$1
shift
clip=$(basename "${video%.*}")
output=outputs
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case ${args[i]} in
    --output)
      if (( i+1 >= ${#args[@]} )) || [[ -z ${args[i+1]} ]]; then echo '--output requires a directory' >&2; exit 2; fi
      output=${args[i+1]}; ((i+=1));;
    --output=*) output=${args[i]#--output=};;
  esac
done
if [[ -z $output ]]; then echo '--output requires a directory' >&2; exit 2; fi
marker=$(mktemp)
trap 'rm -f "$marker"' EXIT
python run_models.py "$video" --no-video --no-per-model "$@"
detections="$output/$clip/detections.json"
if [[ ! -s $detections || ! $detections -nt $marker ]]; then
  echo "Detection stage did not produce fresh results: $detections" >&2
  exit 1
fi
python homography.py "$detections" "$video" --no-video
