#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torchvision import models, transforms
from torchvision import transforms
from torch.nn import functional as F
import json


def cv2_to_tkinter(img):
    """
        cv2をtkinterの形式に変換する関数
    """
    img  = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    img  = Image.fromarray(img) # RGBからPILフォーマットへ変換
    img   = ImageTk.PhotoImage(img ) # ImageTkフォーマットへ変換
    return img

def cv2_to_tensor(img, resize=299):
    """
        cv2をtensorの形式に変換する関数
    """
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
    img = Image.fromarray(img) # RGBからPILフォーマットへ変換
    img = transforms.Resize(resize)(img) # リサイズ
    img = transforms.CenterCrop(resize)(img) # 中心を切り抜く
    img = transforms.ToTensor()(img) # テンソルに変換
    return img

def get_classes():
    """
        ImageNet用のクラスIDとクラス名を参照するためのリストを読み込む関数
    """
    if not os.path.isfile("data/imagenet_class_index.json"):
        # ファイルが存在しない場合はダウンロードする。
        download_url("https://git.io/JebAs", "data", "imagenet_class_index.json")

    # クラス一覧を読み込む。
    with open("data/imagenet_class_index.json", encoding="utf-8") as f:
        data = json.load(f)
        class_names = [x["ja"] for x in data]

    return class_names

# GUI アプリケ-ションを作成するクラス
class ReadImage(tk.Frame):
    def __init__(self,master=None, font='Helvetica', japanese_font="ＭＳ Ｐゴシック", font_size=16):
        super().__init__(master)
        self.font = font
        self.japanese_font = japanese_font
        self.master = master
        self.pack()
        self.screen_rate = self.master.winfo_screenwidth() / 1920 # 画面の解像度の比率 (基準は1920x1080)
        self.font_size = int(font_size * self.screen_rate)
        self.create_widgets()
        self.load_model()
    # widgetsの作成・設定
    def create_widgets(self):
        
        """
            ファイルに関連するウィジェット
        """
        # self.file_fr : ファイルに関連するウィジェットを表示するフレーム
        self.file_fr = tk.Frame(self, bd=2, relief="groove", padx=10, pady=10)
        self.file_fr.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        # self.file_name_up_lb : ファイル名の上に表示するラベル
        self.file_name_up_lb = tk.Label(self.file_fr, text="file name", width=50, font=(self.font,self.font_size))
        self.file_name_up_lb.grid(row=0, column=1)
        # self.width_up_lb : 幅の上に表示するラベル
        self.width_up_lb = tk.Label(self.file_fr, text="Width", width=6, font=(self.font,self.font_size-2))
        self.width_up_lb.grid(row=0, column=2)
        # self.height_up_lb : 高さの上に表示するラベル
        self.height_up_lb = tk.Label(self.file_fr, text="Height", width=6, font=(self.font,self.font_size-2))
        self.height_up_lb.grid(row=0, column=3)
        
        # self.get_btn : ファイルを読み込むボタン
        self.get_btn = tk.Button(self.file_fr,text="open", command=self.get_filepath, font=(self.font,self.font_size))
        self.get_btn.grid(row=1, column=0)
        # self.file_name_lb : 読み込んだファイルの名前を表示するラベル
        self.file_name_lb = tk.Label(self.file_fr, text="", width=38, font=(self.font,self.font_size))
        self.file_name_lb.grid(row=1, column=1)
        # self.width_lb : 読み込んだファイルの幅を表示するラベル
        self.width_lb = tk.Label(self.file_fr, text="", width=6, font=(self.font,self.font_size))
        self.width_lb.grid(row=1, column=2)
        # self.height_lb : 読み込んだファイルの高さを表示するラベル
        self.height_lb = tk.Label(self.file_fr, text="", width=6, font=(self.font,self.font_size))
        self.height_lb.grid(row=1, column=3)
        
        """
            画像に関連するウィジェット
        """
        # self.image_fr : 画像に関連するウィジェットを表示するフレーム
        self.image_fr = tk.Frame(self, bd=2, relief="groove", padx=10, pady=10)
        self.image_fr.grid(row=1, column=0, padx=10, pady=10)
        # self.image_cv : 読み込んだ画像を表示するキャンパス
        self.image_cv = tk.Canvas(self.image_fr,width=int(500*self.screen_rate), height=int(500*self.screen_rate))
        self.image_cv.grid(row=0, column=0)
        
        """
            画像分類に関連するウィジェット
        """
        # self.class_fr : 画像分類に関連するウィジェットを表示するフレーム
        self.class_fr = tk.Frame(self, bd=2, relief="groove", padx=10, pady=10)
        self.class_fr.grid(row=1, column=1, padx=10, pady=10)
        # self.auto_ck : 自動で分類を行うかどうかのチェックボックス
        self.auto_bl = tk.BooleanVar()
        self.auto_bl.set(1) # 初期値はTrue
        self.auto_ck = tk.Checkbutton(self.class_fr, text='Auto Classify',variable=self.auto_bl,
                                      command=self.classify_image, font=(self.font,self.font_size), state="disabled")
        self.auto_ck.grid(row=0,column=0)
        # self.class_bt : 分類を行うボタン
        self.class_bt = tk.Button(self.class_fr,text="Classigy", command=self.classify_image,
                                  font=(self.font,self.font_size), state="disabled")
        self.class_bt.grid(row=0,column=1)
        # self.crop_cv : 分類を行うクロップ画像を表示するキャンバス
        self.crop_cv = tk.Canvas(self.class_fr,width=int(310*self.screen_rate), height=int(310*self.screen_rate))
        self.crop_cv.grid(row=1, column=0, columnspan=2)
        # self.cate_lbs : 分類結果のカテゴリーを表示するラベル
        self.cate_lbs = []
        for i in range(2,7):
            self.cate_lbs.append(tk.Label(self.class_fr, width=25, font=(self.japanese_font,self.font_size-3,"bold"), anchor=tk.E))
            self.cate_lbs[-1].grid(row=i, column=0)
        # self.prob_lbs : 分類結果の確率を表示するラベル
        self.prob_lbs = []
        for i in range(2,7):
            self.prob_lbs.append(tk.Label(self.class_fr, text=":", width=8, font=(self.font,self.font_size), anchor=tk.W))
            self.prob_lbs[-1].grid(row=i, column=1)

        """
            枠操作に関連するウィジェット
        """
        # self.control_fr : 枠操作に関連するウィジェットを表示するフレーム
        self.control_fr = tk.Frame(self, bd=2, relief="groove", padx=10, pady=10)
        self.control_fr.grid(row=2, column=0, sticky=tk.N, padx=10, pady=10)
        # self.control_lb : "Frame Controller"と表示するラベル
        self.control_lb = tk.Label(self.control_fr, text="Frame Controller", font=(self.font,self.font_size))
        self.control_lb.grid(row=0, column=0, columnspan=3)
        
        # 枠のサイズを操作するウィジェット
        # self.size_lb : "size"と表示するラベル
        self.size_lb = tk.Label(self.control_fr, text="size", width=5, font=(self.font,self.font_size), anchor=tk.E)
        self.size_lb.grid(row=1, column=0)
        # self.size_et : 枠サイズを入力できるエントリー
        self.size_et = tk.Entry(self.control_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.size_et.grid(row=1, column=1)
        self.size_et.bind("<Return>",self.rtn_size_et)
        # self.size_sc : 枠のサイズを変更できるスケール
        self.size_var = tk.IntVar()
        self.size_var.set(0)
        self.size_sc = tk.Scale(self.control_fr, from_=1, variable=self.size_var, length=int(360*self.screen_rate), 
                                command=self.upd_size_sc, state="disabled", orient="horizontal", showvalue=False)
        self.size_sc.grid(row=1, column=2)
        
        # 枠のx座標を操作するウィジェット
        # self.x_lb : "x"と表示するラベル
        self.x_lb = tk.Label(self.control_fr, text="x", width=5, font=(self.font,self.font_size), anchor=tk.E)
        self.x_lb.grid(row=2, column=0)
        # self.x_et : 枠のx座標(左上)を入力できるエントリー
        self.x_et = tk.Entry(self.control_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.x_et.grid(row=2, column=1)
        self.x_et.bind("<Return>",self.rtn_loc_et)
        # self.x_sc : 枠のx座標(左上)を変更できるスケール
        self.x_var = tk.IntVar()
        self.x_var.set(0)
        self.x_sc = tk.Scale(self.control_fr, from_=0, variable=self.x_var, length=int(360*self.screen_rate), command=self.upd_loc_sc,
                             state="disabled", orient="horizontal", showvalue=False)
        self.x_sc.grid(row=2, column=2)
        
        # 枠のy座標を操作するウィジェット
        # self.y_lb : "y"と表示するラベル
        self.y_lb = tk.Label(self.control_fr, text="y", width=5, font=(self.font,self.font_size), anchor=tk.E)
        self.y_lb.grid(row=3, column=0)
        # self.y_et : 枠のy座標(左上)を入力できるエントリー
        self.y_et = tk.Entry(self.control_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.y_et.grid(row=3, column=1)
        self.y_et.bind("<Return>",self.rtn_loc_et)
        # self.y_sc : 枠のy座標(左上)を変更できるスケール
        self.y_var = tk.IntVar()
        self.y_var.set(0)
        self.y_sc = tk.Scale(self.control_fr, from_=0, variable=self.y_var, length=int(360*self.screen_rate), command=self.upd_loc_sc, 
                             state="disabled", orient="horizontal", showvalue=False)
        self.y_sc.grid(row=3, column=2)
        
        """
            画像加工に関連するウィジェット
        """
        # self.process_fr : 画像加工に関連するウィジェットを表示するフレーム
        self.process_fr = tk.Frame(self, bd=2, relief="groove", padx=10, pady=10)
        self.process_fr.grid(row=2, column=1, padx=10, pady=10)
        # self.process_lb : "Image Processing"と表示するラベル
        self.process_lb = tk.Label(self.process_fr, text="Image Processing", font=(self.font,self.font_size))
        self.process_lb.grid(row=0, column=0, columnspan=3)
        
        # ぼかし処理を行うウィジェット
        # self.blur_lb : "Blur"と表示するラベル
        self.blur_lb = tk.Label(self.process_fr, text="Blur", width=10, font=(self.font,self.font_size-2), anchor=tk.E)
        self.blur_lb.grid(row=1, column=0)
        # self.blur_et : ぼかしの強さ(カーネルサイズ)を入力できるエントリー
        self.blur_et = tk.Entry(self.process_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.blur_et.grid(row=1, column=1)
        self.blur_et.bind("<Return>",self.rtn_process_et)
        # self.blur_sc : ぼかしの強さ(カーネルサイズ)を変更できるスケール
        self.blur_var = tk.IntVar()
        self.blur_var.set(0)
        self.blur_sc = tk.Scale(self.process_fr, from_=0, variable=self.blur_var, length=int(300*self.screen_rate), 
                                command=self.upd_process_sc, state="disabled", orient="horizontal", showvalue=False)
        self.blur_sc.grid(row=1, column=2)
        
        # モザイク処理を行うウィジェット
        # self.mosaic_lb : "Mosaic"と表示するラベル
        self.mosaic_lb = tk.Label(self.process_fr, text="Mosaic", width=10, font=(self.font,self.font_size-2), anchor=tk.E)
        self.mosaic_lb.grid(row=2, column=0)
        # self.mosaic_et : モザイクの強さを入力できるエントリー
        self.mosaic_et = tk.Entry(self.process_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.mosaic_et.grid(row=2, column=1)
        self.mosaic_et.bind("<Return>",self.rtn_process_et)
        # self.mosaic_sc : モザイクの強さを変更できるスケール
        self.mosaic_var = tk.IntVar()
        self.mosaic_var.set(0)
        self.mosaic_sc = tk.Scale(self.process_fr, from_=0, variable=self.mosaic_var, length=int(300*self.screen_rate), 
                                  command=self.upd_process_sc, state="disabled", orient="horizontal", showvalue=False)
        self.mosaic_sc.grid(row=2, column=2)

        # コントラスト調整を行うウィジェット
        # self.contrast_lb : "Contrast"と表示するラベル
        self.contrast_lb = tk.Label(self.process_fr, text="Contrast", width=10, font=(self.font,self.font_size-2), anchor=tk.E)
        self.contrast_lb.grid(row=3, column=0)
        # self.contrast_et : コントラストの強さを入力できるエントリー
        self.contrast_et = tk.Entry(self.process_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.contrast_et.grid(row=3, column=1)
        self.contrast_et.bind("<Return>",self.rtn_process_et)
        # self.contrast_sc : コントラストの強さを変更できるスケール
        self.contrast_var = tk.IntVar()
        self.contrast_var.set(100)
        self.contrast_sc = tk.Scale(self.process_fr, from_=0, to=200, variable=self.contrast_var, length=int(300*self.screen_rate),
                                    command=self.upd_process_sc, state="disabled", orient="horizontal", showvalue=False)
        self.contrast_sc.grid(row=3, column=2)
        
        # 明るさ調整を行うウィジェット
        # self.bright_lb : "Brightness"と表示するラベル
        self.bright_lb = tk.Label(self.process_fr, text="Brightness", width=10, font=(self.font,self.font_size-2), anchor=tk.E)
        self.bright_lb.grid(row=4, column=0)
        # self.bright_et : 明るさを入力できるエントリー
        self.bright_et = tk.Entry(self.process_fr, width=6, font=(self.font,self.font_size), state="disabled")
        self.bright_et.grid(row=4, column=1)
        self.bright_et.bind("<Return>",self.rtn_process_et)
        # self.bright_sc : 明るさを変更できるスケール
        self.bright_var = tk.IntVar()
        self.bright_var.set(0)
        self.bright_sc = tk.Scale(self.process_fr, from_=-100, to=100, variable=self.bright_var, length=int(300*self.screen_rate), 
                                  command=self.upd_process_sc, state="disabled", orient="horizontal", showvalue=False)
        self.bright_sc.grid(row=4, column=2)
        
        # グレースケール変換を行うウィジェット
        # self.gray_lb : "GrayScale"と表示するラベル
        self.gray_lb = tk.Label(self.process_fr, text="GrayScale", width=10, font=(self.font,self.font_size-2), anchor=tk.E)
        self.gray_lb.grid(row=5, column=0)
        # self.gray_ck : グレースケールを行うかどうかのチェックボックス
        self.gray_bl = tk.BooleanVar()
        self.gray_ck = tk.Checkbutton(self.process_fr, text='',variable=self.gray_bl, command = self.image_processing, state="disabled")
        self.gray_ck.grid(row=5, column=1)

    def load_model(self):
        """
            分類を行うためのモデルを読み込む関数
        """
        # GPUが使える場合はGPUを使用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if int(torchvision.__version__.split('.')[0])>0 or int(torchvision.__version__.split('.')[1])>12:
            # torchvision　0.13以降では引数weightを使用
            self.net = models.inception_v3(weights = "IMAGENET1K_V1").to(self.device)
        else:
            # torchvision　それ以前では引数pretrainedを使用
            self.net = models.inception_v3(pretrained=True).to(self.device)
        self.net.eval()
        # ImageNetのクラス名一覧を取得する。
        self.class_names = get_classes()
        
    def get_filepath(self):
        """"
            self.get_btn(ボタンウィジェット)が押されたときに実行する関数
            画像ファイルを読み込む
        """
        # 読み込む拡張子
        filetype_list = [("Image file", ".bmp .png .jpg .tif .jpeg"), ("all file","*")]
        # ファイルパスを取得
        file_path = filedialog.askopenfilename(initialdir="./", filetypes=filetype_list, title="select file")
        if type(file_path) is str and file_path != "":
            self.original_img = cv2.imread(file_path)
            if self.original_img is not None:
                # control_fr, process_fr, class_frにあるウィジェットをすべてアクティブにする
                widgets_list = self.control_fr.winfo_children()
                widgets_list.extend(self.process_fr.winfo_children())
                widgets_list.extend(self.class_fr.winfo_children())
                for widget in widgets_list:
                    if widget['state'] == "disabled":
                        widget['state'] = "normal"
                # 画像をキャンバスに表示する
                self.resize_img, self.rate = self.show_image(self.original_img, self.image_cv)
                self.img = self.original_img.copy()
                # ファイル名、画像サイズを表示する
                h, w = self.original_img.shape[:2]
                self.original_height, self.original_width = h, w
                self.file_name_lb["text"] = file_path.split('/')[-1]
                self.width_lb["text"] = str(w)
                self.height_lb["text"] = str(h)
                # 枠を作成のコード
                self.frame_max_size = h if h<w else w # 枠の最大値を決定
                self.frame_size = self.frame_max_size # 枠のサイズを最大値で初期化にする
                # 枠が中心になるように枠の左上の座標を計算
                self.frame_x = (w-self.frame_size)/2
                self.frame_y = (h-self.frame_size)/2
                # スケールの最大値を更新
                self.size_sc["to"] = self.frame_max_size # サイズスケールの最大値を変更
                self.x_sc["to"] = self.original_width-self.frame_size # xスケールの最大値を変更
                self.y_sc["to"] = self.original_height-self.frame_size # yスケールの最大値を変更
                self.blur_sc["to"] = int(self.frame_max_size/4) # ぼかしスケールの最大値を変更
                self.mosaic_sc["to"] = int(self.frame_max_size)-1 # モザイクスケールの最大値を変更
                # ウィジェットの値を更新
                self.size_et.delete(0, tk.END)
                self.size_et.insert(tk.END,str(self.frame_size))
                self.size_var.set(self.frame_size)
                self.x_var.set(int(self.frame_x))
                self.x_et.delete(0, tk.END)
                self.x_et.insert(tk.END,str(int(self.frame_x)))
                self.y_var.set(int(self.frame_y))
                self.y_et.delete(0, tk.END)
                self.y_et.insert(tk.END,str(int(self.frame_y)))
                self.blur_et.delete(0, tk.END)
                self.blur_et.insert(tk.END,"0")
                self.mosaic_et.delete(0, tk.END)
                self.mosaic_et.insert(tk.END,"0")
                self.contrast_et.delete(0, tk.END)
                self.contrast_et.insert(tk.END,"1.00")
                self.bright_et.delete(0, tk.END)
                self.bright_et.insert(tk.END,"0")
                self.gray_bl.set(False)
                # 枠を表示するときのオフセットを計算
                if h > w:
                    self.offset_x = (int(self.image_cv["width"])-w*self.rate)/2
                    self.offset_y = 0
                else:
                    self.offset_x = 0
                    self.offset_y = (int(self.image_cv["height"])-h*self.rate)/2
                # 枠のオブジェクトを作成
                self.image_cv.delete("frame")
                self.image_cv.create_rectangle(int(self.frame_x*self.rate)+self.offset_x, # 枠の左上のx座標
                                               int(self.frame_y*self.rate)+self.offset_y, # 枠の左上のy座標
                                               int((self.frame_x+self.frame_size)*self.rate)+self.offset_x, # 枠の右下のx座標
                                               int((self.frame_y+self.frame_size)*self.rate)+self.offset_y, # 枠の右下のy座標
                                               outline='red',width=4, tag="frame")
                self.classify_image() #分類を実行
            else:
                messagebox.showerror("error", "ファイルが読み込みませんでした")
    
    def classify_image(self):
        """
            分類を行う関数
            実行時のself.frame_x, self.frame_y, self.frame_sizeを用いて画像を切り抜き分類する関数
            分類の結果をラベルに表示する
        """
        # クロップする始点と終点を計算
        x1, x2 = int(self.frame_x), int(self.frame_x)+int(self.frame_size)
        y1, y2 = int(self.frame_y), int(self.frame_y)+int(self.frame_size)
        self.crop_img = self.img[y1:y2,x1:x2]
        # クロップ画像を表示
        self.crop_resize_img, _ = self.show_image(self.crop_img, self.crop_cv)
        inputs = cv2_to_tensor(self.crop_img).unsqueeze(0) # テンソルに変換
        inputs = inputs.to(self.device) 
        outputs = self.net(inputs) # 分類モデルにクロップ画像を入力
        outputs = F.softmax(outputs, dim=1) 
        probs, indices = outputs.cpu().sort(dim=1, descending=True) # 高い順にソート
        for i in range(5):
            self.cate_lbs[i]["text"] = self.class_names[indices[0][i]]
            self.prob_lbs[i]["text"] = ": {:.5f}".format(probs[0][i])
    
    def show_image(self,image,canvas):
        """"
            画像をキャンバスウィジェットに合わせて表示を行う関数
            
            image  : 表示する画像 (cv2)
            canvas : 表示するキャンバスウィジェット (tk.Canvas)
        """
        # キャンバスに合わせてリサイズする高さと幅を計算
        h, w = image.shape[:2]
        cv_h, cv_w = int(canvas['height']), int(canvas['width'])
        if h > w: # 画像が縦長い場合、高さに合わせる
            rate = cv_h / h
        else:     # 画像が横長い場合、幅に合わせる
            rate = cv_w / w
        resize_img = cv2.resize(image, dsize=(int(w*rate), int(h*rate)))
        # ImageTkフォーマットへ変換
        resize_img = cv2_to_tkinter(resize_img)
        canvas.delete("image")
        canvas.create_image(int(cv_w/2), int(cv_h/2), image=resize_img , anchor='center', tag="image")
        canvas.lower("image")
        return resize_img, rate
        
    def rtn_size_et(self, event=None):
        """
            self.size_etでReturnを押すと呼び出される関数
        """
        try:
            if int(self.size_et.get()) < 1: # 0以下が入力された場合、1にする
                self.size_et.delete(0, tk.END)
                self.size_et.insert(tk.END,"1")
            elif int(self.size_et.get()) > self.frame_max_size: # 最大値以上が入力された場合、最大値にする
                self.size_et.delete(0, tk.END)
                self.size_et.insert(tk.END,str(self.frame_max_size))
            self.size_var.set(int(self.size_et.get())) # スケールを更新
            self.change_size() # 枠のサイズを変更
        except:
            messagebox.showerror("error", "数字を入力してください")
            
    def upd_size_sc(self, event=None):
        """
            self.size_scを動かすたびに呼び出される関数
        """
        self.change_size() # 枠のサイズを変更
        # size_etウィジェットの値を更新
        self.size_et.delete(0, tk.END)
        self.size_et.insert(tk.END,str(self.frame_size))
        
    def change_size(self):
        """
            枠のサイズを変更する関数
            実行時のself.size_varの値を用いて処理を行う
        """
        change_rate = int(self.size_var.get()) / self.frame_size # サイズを変更する割合を計算
        # 枠の中心座標を計算 (キャンバスの座標)
        center_x = (self.frame_x+self.frame_size/2)*self.rate+self.offset_x
        center_y = (self.frame_y+self.frame_size/2)*self.rate+self.offset_y
        # 枠のサイズを変更
        self.image_cv.scale("frame", center_x, center_y, change_rate, change_rate) 
        self.frame_size = int(self.size_var.get()) # 枠のサイズの値を更新
        self.x_sc["to"] = self.original_width-self.frame_size # xスケールの最大値を変更
        self.y_sc["to"] = self.original_height-self.frame_size # yスケールの最大値を変更
        # 枠の左上の座標を計算 (画像内の座標)
        self.frame_x = (self.image_cv.coords("frame")[0]-self.offset_x)/self.rate
        self.frame_y = (self.image_cv.coords("frame")[1]-self.offset_y)/self.rate
        # 枠が画像内に収まるように移動
        max_x = self.original_width-self.frame_size
        max_y = self.original_height-self.frame_size
        if self.frame_x < 0:
            self.image_cv.move("frame", (0-self.frame_x)*self.rate, 0)
            self.frame_x = 0
        elif self.frame_x > max_x:
            self.image_cv.move("frame", (max_x-self.frame_x)*self.rate, 0)
            self.frame_x = max_x
        if self.frame_y < 0:
            self.image_cv.move("frame", 0, (0-self.frame_y)*self.rate)
            self.frame_y = 0
        elif self.frame_y > max_y:
            self.image_cv.move("frame", 0, (max_y-self.frame_y)*self.rate)
            self.frame_y = max_y
        # ウィジェットを座標の値に更新
        self.x_var.set(int(self.frame_x))
        self.x_et.delete(0, tk.END)
        self.x_et.insert(tk.END,str(int(self.frame_x)))
        self.y_var.set(int(self.frame_y))
        self.y_et.delete(0, tk.END)
        self.y_et.insert(tk.END,str(int(self.frame_y)))
        if self.auto_bl.get(): # self.auto_ckにチェックが入っているなら
            self.classify_image() # サイズ変更後に分類を行う
        
    
    def rtn_loc_et(self, event=None):
        """
            self.x_et, self.y_etでReturnを押すと呼び出される関数
        """
        try:
            # x座標の最小値, 最大値を設定
            if int(self.x_et.get()) < 0: # -1以下が入力された場合、0にする
                self.x_et.delete(0, tk.END)
                self.x_et.insert(tk.END,"0")
            elif int(self.x_et.get()) > self.original_width-self.frame_size: # 最大値以上が入力された場合、最大値にする
                self.x_et.delete(0, tk.END)
                self.x_et.insert(tk.END,str(self.original_width-self.frame_size))
            # y座標の最小値, 最大値を設定
            if int(self.y_et.get()) < 0: # -1以下が入力された場合、0にする
                self.y_et.delete(0, tk.END)
                self.y_et.insert(tk.END,"0")
            elif int(self.y_et.get()) > self.original_height-self.frame_size: # 最大値以上が入力された場合、最大値にする
                self.y_et.delete(0, tk.END)
                self.y_et.insert(tk.END,str(self.original_height-self.frame_size))
            self.x_var.set(int(self.x_et.get())) # x座標のスケールを更新
            self.y_var.set(int(self.y_et.get())) # y座標のスケールを更新
            self.move_frame() # 枠を移動
        except:
            messagebox.showerror("error", "数字を入力してください")
    
    def upd_loc_sc(self, event=None):
        # x座標のエントリーウィジェットの値を更新
        self.x_et.delete(0, tk.END)
        self.x_et.insert(tk.END,str(int(self.frame_x)))
        # y座標のエントリーウィジェットの値を更新
        self.y_et.delete(0, tk.END)
        self.y_et.insert(tk.END,str(int(self.frame_y)))
        self.move_frame() # 枠を移動
        
    def move_frame(self):
        """
            枠を移動させる関数
            実行時のスケールの値を用いて処理を行う
        """
        move_x = (int(self.x_var.get())-self.frame_x)*self.rate # x座標の移動量を計算 (キャンバスの座標)
        move_y = (int(self.y_var.get())-self.frame_y)*self.rate # y座標の移動量を計算 (キャンバスの座標)
        self.image_cv.move("frame", move_x, move_y) # 枠を移動
        self.frame_x = int(self.x_var.get()) # 移動後にx座標の値を更新
        self.frame_y = int(self.y_var.get()) # 移動後にy座標の値を更新
        if self.auto_bl.get(): # self.auto_ckにチェックが入っているなら
            self.classify_image() # 移動後に分類を行う
            
    def rtn_process_et(self, event=None):
        """
            画像加工に関するエントリーウィジェット
            self.blur_et, self.mosaic_et, self.contrast_et, self.bright_et
            でReturnを押すと呼び出される関数
        """
        try:
            # blur_etの最小値、最大値を設定
            if int(self.blur_et.get()) < 0: # 0より小さい値が入力された場合、0にする
                self.blur_et.delete(0, tk.END)
                self.blur_et.insert(tk.END,"0")
            elif int(self.blur_et.get()) > self.blur_sc["to"]: # 最大値より大きい値が入力された場合、最大値にする
                self.blur_et.delete(0, tk.END)
                self.blur_et.insert(tk.END,str(int(self.blur_sc["to"])))
            # mosaic_etの最小値、最大値を設定
            if int(self.mosaic_et.get()) < 0: # 0より小さい値が入力された場合、0にする
                self.mosaic_et.delete(0, tk.END)
                self.mosaic_et.insert(tk.END,"0")
            elif int(self.mosaic_et.get()) > self.mosaic_sc["to"]: # 最大値より大きい値が入力された場合、最大値にする
                self.mosaic_et.delete(0, tk.END)
                self.mosaic_et.insert(tk.END,str(int(self.mosaic_sc["to"])))
            # contrast_etの最小値、最大値を設定
            if float(self.contrast_et.get()) < 0: # 0より小さい値が入力された場合、0.00にする
                self.contrast_et.delete(0, tk.END)
                self.contrast_et.insert(tk.END,"0.00")
            elif float(self.contrast_et.get()) > 2: # 2より大きい値が入力された場合、2.00にする
                self.contrast_et.delete(0, tk.END)
                self.contrast_et.insert(tk.END,"2.00")
            # bright_etの最小値、最大値を設定
            if int(self.bright_et.get()) < -100: # -100より小さい値が入力された場合、-100にする
                self.bright_et.delete(0, tk.END)
                self.bright_et.insert(tk.END,"-100")
            elif int(self.bright_et.get()) > 100: # 100より大きい値が入力された場合、100にする
                self.bright_et.delete(0, tk.END)
                self.bright_et.insert(tk.END,"100")
            # スケールを更新
            self.blur_var.set(int(self.blur_et.get())) # ぼかしスケールを更新
            self.mosaic_var.set(int(self.mosaic_et.get())) # モザイクスケールを更新
            self.contrast_var.set(float(self.contrast_et.get())*100) # コントラストスケールを更新
            self.bright_var.set(int(self.bright_et.get())) # 明るさスケールを更新
            # 画像加工処理を行う
            self.image_processing()
        except:
            messagebox.showerror("error", "数字を入力してください")
            
    def upd_process_sc(self,event=None):
        """
            画像加工に関するスケールウィジェット
            self.blur_sc, self.mosaic_sc, self.contrast_sc, self.bright_sc
            を動かすたびに呼び出される関数
        """
        # エントリーウィジェットの値を更新
        self.blur_et.delete(0, tk.END)
        self.blur_et.insert(tk.END,self.blur_var.get())
        self.mosaic_et.delete(0, tk.END)
        self.mosaic_et.insert(tk.END,self.mosaic_var.get()) 
        self.contrast_et.delete(0, tk.END)
        self.contrast_et.insert(tk.END,'{:.2f}'.format(float(self.contrast_var.get())/100)) 
        self.bright_et.delete(0, tk.END)
        self.bright_et.insert(tk.END,self.bright_var.get()) 
        # 画像加工処理を行う
        self.image_processing()

    def image_processing(self):
        """
            画像の加工を行い表示する関数
            実行時のスケールの値を用いて処理を行う
        """
        # コントラスト、明るさ調整を行う
        self.img = float(self.contrast_var.get())*self.original_img/100 + int(self.bright_var.get())
        self.img = np.clip(self.img,0,255).astype(np.uint8)
        # ガウシアンフィルタでぼかす
        kernel = (int(self.blur_var.get())*2+1,int(self.blur_var.get())*2+1)
        self.img = cv2.GaussianBlur(self.img,kernel,0)
        # 縮小→拡大によってモザイク処理を行う
        ratio = (self.frame_max_size-int(self.mosaic_var.get()))/self.frame_max_size
        small_img = cv2.resize(self.img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
        self.img = cv2.resize(small_img, self.original_img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        # グレースケール変換を行う
        if self.gray_bl.get():
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 加工後の画像を表示
        self.resize_img, _ = self.show_image(self.img, self.image_cv)
        if self.auto_bl.get(): # self.auto_ckにチェックが入っているなら
            self.classify_image() # 加工後に分類を行う


if __name__ == "__main__": 
    root = tk.Tk()
    root.title("GUI_Classifier")
    app = ReadImage(master=root)
    app.mainloop()

