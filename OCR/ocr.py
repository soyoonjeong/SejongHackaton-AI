#!pip install easyocr
#!pip install pdf2image pillow
#!sudo apt-get install -y poppler-utils

import os
from pdf2image import convert_from_path
import os
import easyocr

def pdftoimage(pdf_path, save_dir, dpi=300):
    """
    PDF 파일을 이미지로 변환하여 저장하는 함수

    Parameters
    ----------
    pdf_path : str
        변환할 PDF 파일 경로
    save_dir : str
        변환된 이미지를 저장할 디렉토리 경로
    dpi : int, optional
        이미지 해상도 (기본값 300)
    """

    # 디렉토리 생성 (존재하지 않으면 생성)
    os.makedirs(save_dir, exist_ok=True)

    # PDF를 이미지로 변환
    images = convert_from_path(pdf_path, dpi=dpi)

    # 변환된 이미지를 지정된 경로에 저장
    for i, image in enumerate(images):
        image_path = os.path.join(save_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        print(f"Saved: {image_path}")



def ocr_images_to_single_txt(image_dir, output_txt_path, lang_list=['en', 'ko'], use_gpu=False):
    """
    주어진 디렉토리 내의 모든 PNG 이미지를 EasyOCR로 인식하고,
    결과를 하나의 txt 파일로 저장하는 함수

    Parameters
    ----------
    image_dir : str
        OCR을 수행할 이미지들이 위치한 디렉토리 경로
    output_txt_path : str
        모든 OCR 결과를 기록할 텍스트 파일 경로
    lang_list : list, optional
        OCR에서 사용할 언어 리스트 (기본값 ['en', 'ko'])
    use_gpu : bool, optional
        GPU 사용 여부 (기본값 False)
    """

    # OCR Reader 초기화
    reader = easyocr.Reader(lang_list, gpu=use_gpu)

    # 디렉토리 내 모든 PNG 파일 목록 생성
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    if not image_files:
        print(f"'{image_dir}' 디렉토리 내에 PNG 파일이 없습니다.")
        return

    # 하나의 텍스트 파일에 순차적으로 작성하기 위해 'w' 모드로 열기
    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
        # 각 이미지에 대해 OCR 수행
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"Processing: {image_path}")

            # OCR 수행
            result = reader.readtext(image_path)

            # 이미지 구분을 위해 이미지 파일명 삽입
            #out_file.write(f"=== {image_file} ===\n")

            # OCR 결과를 txt 파일에 기록
            for (bbox, text, prob) in result:
                if prob < 0.5:
                    continue  # 신뢰도 50% 미만은 무시
                out_file.write(f"{text}\n")
                # 추가 정보 기록 예시(주석 해제):
                # out_file.write(f"Detected text: {text}\nBounding Box: {bbox}\nConfidence: {prob:.2f}\n\n")

            out_file.write("\n")  # 이미지별 공백 라인 삽입
            print(f"Finished OCR for: {image_file}")
            print("-" * 50)

    print(f"모든 OCR 결과가 '{output_txt_path}' 파일에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    pdf_path = "/content/drive/MyDrive/OCR_data/pdf/[교안] 정보보호와 보안의 기초 2주차 1강.pdf"
    save_dir = "/content/drive/MyDrive/OCR_data/image_정보보호"

    pdftoimage(pdf_path, save_dir)
    
    image_dir = "/content/drive/MyDrive/OCR_data/image_정보보호"
    output_txt_path = "/content/drive/MyDrive/OCR_data/all_ocr_results.txt"

    ocr_images_to_single_txt(
        image_dir=image_dir,
        output_txt_path=output_txt_path,
        lang_list=['en', 'ko'],
        use_gpu=True
    )