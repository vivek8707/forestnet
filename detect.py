import cv2
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

credentials = ApiKeyCredentials(in_headers = {"Prediction-Key":"3e6e3b743d744b93bf0b30d5c9d5a391"})
predictor =  CustomVisionPredictionClient("https://southcentralus.api.cognitive.microsoft.com/", credentials)
def predict(path):
    
    img = cv2.imread(path)
    img_shape = img.shape
    print(img_shape)
    with open(path, mode ='rb') as captured_image:

        print("load image... and predict ")
        results = predictor.detect_image("1de5f3a4-6330-41dc-8335-d5e3dd167236", "Iteration1", captured_image)
        # print(results)
        print(results)
        
        for prediction in results.predictions:
            
            if prediction.probability >=0.8:
                print("\t" + prediction.tag_name +": {0:.2f}%".format(prediction.probability * 100))
                left = prediction.bounding_box.left*img.shape[1]
                top= prediction.bounding_box.top*img.shape[0]
                width = prediction.bounding_box.width*img.shape[1]
                height = prediction.bounding_box.height*img.shape[0]
                print( (int(left), int(top)), ( int(top+height), int(left+width)))
                cv2.rectangle(img, ( int(left), int(top)), ( int(left+width), int(top+height)), (0,0,255),1)
                
                print(prediction.bounding_box.left)
                print(prediction.bounding_box.width)
                print(prediction.bounding_box.height)
                print(prediction.bounding_box)
        captured_image.close()
        cv2.imwrite( "./static/assets/img/output2.png", img)

                
            
        cv2.imshow("frame", img)   
        # cv2.waitKey(0)
                
        return results
# predict("./static/assets/img/image_name.jpg")
print("all done")