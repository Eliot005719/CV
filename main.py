from tkinter import *
import tkinter as tk
from tkinter import ttk
import requests
import os
from tkinter import filedialog

 ##https://www.tutorialspoint.com/python_blockchain/python_blockchain_tutorial.pdf
class downloader:
    def __init__(self):
        self.saveto=""
        self.window = tk.Tk()
        self.window.title("Python GUI downloader")
        self.url_label = tk.Label(text="enter url")
        self.url_label.pack()
        self.url_entry= tk.Entry()
        self.url_entry.pack()
        self.browse_button = tk.Button(text="Browse",command=self.browse_file)
        self.browse_button.pack()
        self.download_button=tk.Button(text='download',command=self.download)
        self.download_button.pack()
        self.window.geometry("884x344")
        self.progress_bar =ttk.Progressbar(self.window,orient="horizontal",maximum=100,length=300,mode="determinate")
        self.progress_bar.pack()
        self.window.mainloop()

    
    def browse_file(self):
        saveto=filedialog.asksaveasfilename(initialfile="this.pdf")
        self.saveto=saveto


    def download(self):
      url=self.url_entry.get()
      response=requests.get(url,stream=True)
      total_size_in_bytes= int(response.headers.get("Content-length"))
      block_siize=10000
      self.progress_bar["value"]=0
      fileName=self.url_entry.get().split("/")[-1]
      if self.saveto is not"":
         fileName=os.path.join(self.saveto)
         print(fileName)
      with open(fileName,"wb") as f:
          for data in response.iter_content(block_siize):
           self.progress_bar["value"] += (100*block_siize)/total_size_in_bytes
           print(self.progress_bar['value'])
           self.window.update()
           f.write(data)




downloader()
