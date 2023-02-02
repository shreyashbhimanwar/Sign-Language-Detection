import cv2, os

def flip_images():
	gest_folder = "NewGestures"
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir(gest_folder):
		for i in range(300):
			path = gest_folder+"/"+g_id+"/"+str(i+300)+".jpg"
			new_path = gest_folder+"/"+g_id+"/"+str(i+1+600)+".jpg"
			print(path)
			img = cv2.imread(path, 0)
			img = cv2.flip(img, 1)
			cv2.imwrite(new_path, img)

flip_images()
