for file in ../dataset/Validation/video/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../dataset/Validation/audio/${filename}.mp4.wav"
done