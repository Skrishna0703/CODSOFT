import re
import tkinter as tk
from tkinter import scrolledtext

def simple_chatbot(user_input):
   # Convert user input to lowercase for case-insensitivity
    user_input = user_input.lower()

    # Define rules and responses
    greetings = ['hello', 'hi', 'hey', 'greetings']
    farewells = ['bye', 'goodbye', 'see you', 'farewell']
    gm = ['good morning']
    name = ['who are you','what is your name','your name']
    dev=['who created you','who made you']

    # Check user input against predefined rules
    if any(greeting in user_input for greeting in greetings):
        return "Hello! How can I help you today?"

    elif any(farewell in user_input for farewell in farewells):
        return "Goodbye! Have a great day."

    elif any(gm in user_input for gm in gm):
        return "Good Morning Sir/Mam...!"
    
    elif any(name in user_input for name in name):
        return "I'm AI Chatbot.My Name is Wednesday...!"

    elif any(dev in user_input for dev in dev):
        return "Created by Shrikrishna Sutar"

    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    for rule in rules:
        if re.search(rule["pattern"], user_input, re.IGNORECASE):
            return rule["response"]

class ChatbotUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Chatbot")

        self.create_widgets()

    def create_widgets(self):
        self.chat_area = scrolledtext.ScrolledText(self.root, width=50, height=20, wrap=tk.WORD)
        self.chat_area.grid(row=0, column=0, padx=10, pady=10)

        self.user_input_entry = tk.Entry(self.root, width=40)
        self.user_input_entry.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

    def send_message(self):
        user_input = self.user_input_entry.get()
        self.chat_area.insert(tk.END, f"You: {user_input}\n")
        response = simple_chatbot(user_input)
        self.chat_area.insert(tk.END, f"Bot: {response}\n")
        self.user_input_entry.delete(0, tk.END)  # Clear the input field

    def run(self):
        self.root.mainloop()

# Create an instance of the ChatbotUI class and run the UI
chatbot_ui = ChatbotUI()
chatbot_ui.run()
