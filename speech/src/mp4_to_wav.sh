for file in ../Validation/Videos/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../Validation/audio/${filename}.mp4.wav"
done

for file in ../Training/Videos/*.mp4; do
    filename=$(basename "$file" .mp4)
    ffmpeg -i "$file" -vn -ac 1 -ar 16000 -acodec pcm_s16le "../Training/audio/${filename}.mp4.wav"
done