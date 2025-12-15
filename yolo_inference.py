from ultralytics import YOLO

model = YOLO(r"C:\Users\emade\Downloads\AI Track\NTI_Tasks\CV\football_project\models\best.pt")

results = model.predict('input_videos\input_videos.mp4', save=True)
print(results[0])

print("*********************************")
for box in results[0].boxes:  
   print(box)