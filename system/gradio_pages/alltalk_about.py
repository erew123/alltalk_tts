import gradio as gr

def alltalk_about():
    about = """
    ### 🛠️ About this project
    > AllTalk is a labour of love that has been developed, supported and sustained in my personal free time. As a solo enthusiast (not a business or team) my resources are inherently limited. This project has been one of my passions, but I must balance it with other commitments.<br><br>
    To manage AllTalk sustainably, I prioritize support requests based on their overall impact and the number of users affected. I encourage you to utilize the comprehensive documentation and engage with the AllTalk community discussion area. These resources often provide immediate answers and foster a supportive user network.<br><br>
    Should your inquiry extend beyond the documentation, especially if it concerns a bug or feature request, I assure you I’ll offer my best support as my schedule permits. However, please be prepared for varying response times, reflective of the personal dedication I bring to AllTalk. Your understanding and patience in this regard are greatly appreciated.<br><br>
    It's important to note that **I am not** the developer of any TTS models utilized by AllTalk, nor do I claim to be an expert on them, including understanding all their nuances, issues, and quirks. For specific TTS model concerns, I’ve provided links to the original developers in each TTS engines "Engine Information" section for direct assistance/research.<br><br>
    Thank you for your continued support and understanding.
    
    ### 💖 Showing Your Support
    > If AllTalk has been helpful to you, consider showing your support through a donation on my **[Ko-fi page](https://ko-fi.com/erew123)**. <br><br>
    Your support is greatly appreciated and helps ensure the continued development and improvement of AllTalk.
    """
    gr.Markdown(about)