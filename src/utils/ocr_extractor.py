from curses.ascii import isdigit
import easyocr, os


class OCRExtractor:
    """class to extract text from image using OCR
    functions:
        extract_single_image: extract text from a single image
        extract_multiple_images: extract text from multiple images
        __init__: initialize OCR reader
    """
    def __init__(self, gpu_state=False):
        """initialize OCR reader

        Args:
            gpu_state (bool, optional): whether to use GPU or not.
            Defaults to False.
        """
        self.reader = easyocr.Reader(['ch_sim','en'], gpu=gpu_state)

    def modify_extracted_sentences(self, list_text):
        """modify extracted sentences based on certain rules

        Args:
            list_text (list): list of extracted sentences

        Returns:
            list: list of modified sentences
        """
        for sentence_index in range(len(list_text)):

            # replace 夭 with 天
            list_text[sentence_index] = list_text[sentence_index].replace("夭", "天")

            # replace 8 with 日 if the previous two characters are digits
            for char_index in range(len(list_text[sentence_index])):
                if list_text[sentence_index][char_index] == "8" and char_index >= 2:
                    if isdigit(list_text[sentence_index][char_index-1]) and isdigit(list_text[sentence_index][char_index-2]):
                        list_text[sentence_index] = str(list_text[sentence_index][:char_index] + "日" + list_text[sentence_index][char_index+1:])
            
        actual_list_text = []

        # if a sentence does not end with a 。,
        # and the next sentence starts with a Chinese character or English character,
        # then merge them, and restart the loop
        current_sentence = list_text[0]
        for sentence_index in range(len(list_text)):
            if current_sentence[-1] != "。":
                if sentence_index != len(list_text)-1:
                    sent = list_text[sentence_index+1]
                    if len(sent) == 0: continue
                    if sent.isalpha() or (sent[0] >= u'\u4e00' and sent[0] <= u'\u9fa5'):
                        current_sentence += sent
                        continue
            actual_list_text.append(current_sentence)
            if sentence_index != len(list_text)-1:
                current_sentence = list_text[sentence_index+1]

        return actual_list_text

    def extract_single_image(self, image):
        """extract text from image

        Args:
            image (path to image / OpenCV image object / an image file as bytes): 
            image to be extracted

        Returns:
            list: list of text
        """
        list_text = self.reader.readtext(image, detail=0, paragraph=True)
        if list_text:
            list_text = self.modify_extracted_sentences(list_text)
        return list_text

    def extract_multiple_images(self, images):
        """extract text from multiple images

        Args:
            images (list): list of images to be extracted, 
            each image is in the same form as image in extract_single_image

        Returns:
            images_text_list(list): list of listed text extracted from images
        """
        images_text_list = []
        for image in images:
            current_list = self.modify_extracted_sentences(self.extract_single_image(image))
            images_text_list.append(current_list)
        return images_text_list

    def demo(self):
        """demo of the class
        """
        test_ocr = "/root/work/Tsingenda-backend/data/test_ocr"

        print("extracting text from single image, trying test1_CET\n")
        print(self.extract_single_image(os.path.join(test_ocr, "test1_CET.png")))
        print("\nextracting text from multiple images, trying test1_CET and test2_PCR\n")
        print(self.extract_multiple_images([os.path.join(test_ocr, "test1_CET.png"), os.path.join(test_ocr, "test2_PCR.png")]))
        print("\ndemo finished\n")


if __name__ == "__main__":
    ocr = OCRExtractor(gpu_state = True)
    ocr.demo()