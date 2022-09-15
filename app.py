from flask import Flask, render_template, send_file, request
import cv2
import numpy as np
import pandas as pd
import os
import xlsxwriter
from pdf2image import convert_from_path
from detectTable.detectTable import extractTable
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
config = Cfg.load_config_from_name('vgg_seq2seq')

config['cnn']['pretrained']=False
config['device'] = 'cpu'
config['predictor']['beamsearch']=False
detector = Predictor(config)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/', methods=['POST', 'GET'])
def index():
    # Nếu là POST (gửi file)
    if request.method == "POST":
        try:
            # Lấy file gửi lên
            pdf_file = request.files['file']
            if pdf_file:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                print("Save = ", path_to_save)
                pdf_file.save(path_to_save)
                # convert pdf to img
                if('/' in path_to_save):
                    arr1=path_to_save.split('/')
                else:
                    arr1=path_to_save.split('\\')
                fname = arr1[len(arr1)-1].replace(".pdf", ".xlsx")
                pages = convert_from_path(path_to_save)
                if os.path.exists('static/input/') == False:
                    os.mkdir('static/input/')

                no_page = len(pages)
                dim = (984, 693)
                for i in range(3):
                    pages[i] = pages[i].resize(dim)
                    pages[i].save('static/input/page'+ str(i) +'.jpg', 'JPEG')
                if os.path.exists('static/output/') == False:
                    os.mkdir('static/output/')
                writer = pd.ExcelWriter('static/output/result.xlsx', engine='xlsxwriter')
                for i in range(3):
                    path = "static/input/page" + str(i) +'.jpg'
                    img = cv2.imread(path,0)
                    data = extractTable(img, detector)
                    data.to_excel(writer, sheet_name='Sheet'+ str(i))
                writer.save()
                return render_template("index.html", excell_file = fname, msg="Tải file lên thành công")

            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')
@app.route('/download')
def download_file():
    p = "static/output/result.xlsx"
    return send_file(p, as_attachment=True)
if __name__ == '__main__':
    app.debug = False
    app.run()