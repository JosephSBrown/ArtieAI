from cgitb import text
import tkinter as tk
from tkinter import CENTER, END, font

from chat import bot_name, bot_response

background = '#121212'
foreground = '#FFFFFF'
chatbg = '#1F1B24'

class Application:
    def __init__(self):
        self.app = tk.Tk()
        self.mainwindow()

    def Run(self):
        self.app.mainloop()

    def mainwindow(self):
        main = self.app
        main.title("Artie AI | Artificially Intelligent Chat Bot")
        main.resizable(width=False, height=False)
        main.configure(width=500, height=700, bg=background)

        pagetitle = tk.Label(main, text="WELCOME TO ARTIE AI", bg=background, fg=foreground, pady=10, font=("Rockwell", "20", "bold"))
        Explain = tk.Label(main, text="I'm Artie, an Artificially Intelligent Chatbot\nConverse With Me Whenever You're Ready :)", bg=background, fg=foreground, font=("Rockwell", "12"))

        self.DisplayText = tk.Text(main, width=20, height=2, bg=chatbg, fg=foreground, padx=5, pady=5, font=("Rockwell", "12"))
        self.DisplayText.configure(cursor="arrow", state=tk.DISABLED)
        Scroll = tk.Scrollbar(self.DisplayText)
        Scroll.place(relheight=1, relx=0.974)
        Scroll.configure(command=self.DisplayText.yview)

        self.TextInput = tk.Entry(bg=chatbg, fg=foreground, font=("Rockwell", "12"))
        self.TextInput.place(relwidth=0.74, relheight=0.06, rely=0.9, relx=0.02)
        self.TextInput.focus()
        self.TextInput.bind("<Return>", self.EnterPress)

        enter = tk.Button(main, text="SEND", bg=chatbg, fg=foreground, font=("Rockwell", "12", 'bold'), command=lambda: self.EnterPress(None))
        enter.place(rely=0.9, relx=0.8, relheight=0.062, relwidth=0.18)

        clear = tk.Button(main, text="Clear", bg=chatbg, fg=foreground, command=self.clear, font=("Rockwell", "12", "bold"))
        clear.place(relx=0.85, rely=0.04, relwidth=0.12)

        #darkmode = tk.Button(main, text="Toggle\n☽/☼", bg=chatbg, fg=foreground, font=("Rockwell", "12", "bold"))
        #darkmode.place(relx=0.03, rely=0.025, relwidth=0.14)

        pagetitle.place(relwidth=1)
        Explain.place(relwidth=1, rely=0.06, relheight=0.06)
        self.DisplayText.place(relwidth=0.98, relheight=0.745, rely=0.125, relx=0.01)

        software = tk.Label(main, text="Artie, Aritifically Intelligent Chatbot | Developed By Joseph Brown 2022", fg=foreground, bg=background, font=("Rockwell", "8", "italic"))
        software.place(rely=0.97, relwidth=1)
    
    def EnterPress(self, event):
        text = self.TextInput.get()
        self.TextInsert(text, "You")

    def TextInsert(self, text, sender):
        if not text:
            return
        
        self.TextInput.delete(0, END)
        message = f"{sender}: {text}\n"
        self.DisplayText.configure(state=tk.NORMAL)
        self.DisplayText.insert(END, message)
        self.DisplayText.configure(state=tk.DISABLED)

        self.Artie()

    def clear(self):
        self.DisplayText.configure(state=tk.NORMAL)
        self.DisplayText.delete(1.0, END)
        self.DisplayText.configure(state=tk.DISABLED)

    def Artie(self):
        self.TextInput.delete(0, END)
        message = f"{bot_name}: {bot_response}\n"
        self.DisplayText.configure(state=tk.NORMAL)
        self.DisplayText.insert(END, message)
        self.DisplayText.configure(state=tk.DISABLED)

    #def darkmode(self):
        #pass

if __name__ == '__main__':
    App = Application()
    App.Run()

