import whisper
import json
import os


# Load Whisper model
model= whisper.load_model("large-v2")

# List all files in audio folder
audios=os.listdir("audio")

for audio in audios:

    base = os.path.splitext(audio)[0]   # .mp3 hata
    number, title = base.split(" ", 1)  # Lec-1 | title

    print(number, title)
    result=model.transcribe(audio=f"audio/{audio}",
                            language="hi",
                             task="translate",
                             word_timestamps=False)
    
    chunks=[]
    for segment in result['segments']:
        chunks.append({"number": number,"title": title, "start": segment['start'],"end": segment['end'], 
                          "text": segment['text']})
        
    chunks_with_metadata={"chunks":chunks,"text": result['text']}

    with open(f"jsons/{audio}.json","w",encoding="utf-8") as f:
        json.dump(chunks_with_metadata,f,ensure_ascii=False)




# import whisper
# import json
# import os

# # 1️⃣ Load model (CPU pe heavy hai, but tum chala rahi ho)
# model = whisper.load_model("large-v2")

# # 2️⃣ Sample audio filename
# audio = "Lec-19 sample of 🌲 lec.mp3"
# audio_path = f"audio/{audio}"

# # 3️⃣ Extract number & title safely (emoji + spaces supported)
# base = os.path.splitext(audio)[0]   # remove .mp3
# number, title = base.split(" ", 1)  # Lec-19 | sample of 🌲 lec

# print("Number:", number)
# print("Title:", title)

# # 4️⃣ Transcribe ONLY this sample audio
# result = model.transcribe(
#     audio=audio_path,
#     language="hi",
#     task="translate",
#     word_timestamps=False
# )

# # 5️⃣ Build chunks
# chunks = []
# for segment in result["segments"]:
#     chunks.append({
#         "number": number,
#         "title": title,
#         "start": segment["start"],
#         "end": segment["end"],
#         "text": segment["text"]
#     })

# # 6️⃣ Final JSON structure
# output = {
#     "chunks": chunks,
#     "text": result["text"]
# }

# # 7️⃣ Save JSON
# with open("jsons/sample.json", "w",encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False)

