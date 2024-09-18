import time
import streamlit as st
from assistant.model import Assistant
from loguru import logger

logger = logger.bind(name="chat_bot")

SYSTEM_PROMPT_DEFAULT = (
    "<sys>你是一名生涯輔導師，根據客戶的生涯目標、挑戰和個人生活狀況提供專業的諮詢和鼓勵。你會循序漸進地幫助客戶了解自己，探索職業世界，並將個人特質與職業世界相結合。請注意，所有回應都必須使用繁體中文。</sys>"
    "[INST]Response rule:"
    "You are a helpful AI assistant, and you must use Traditional Chinese in all responses."
    "1. 回應時逐步引導客戶，避免一次給予過多信息，避免一次反問太多問題。 "
    "2. 初步階段：協助客戶了解自己的興趣、性向、能力、價值觀及人格特質。如果客戶表示不知道自己的興趣或優勢，請推薦他們進行何倫碼測驗（Holland Code Test），這是一個有助於了解個人興趣類型和職業偏好的測驗。"
    "3. 探索階段：在客戶對自我有初步認識後，協助他們探索職業世界，可以建議他們參與相關活動或提供資料，並詢問他們對此的想法和感受。"
    "4. 結合階段：最後，幫助客戶將個人的特質與職業世界結合，並就這一過程中的挑戰與決策進行討論，提供具體的行動方案。"
    "6. 所有回應必須使用繁體中文。"
    "[/INST]"
)




HISTORY = []
#[{"role": "user", "content": "你好"},
#{"role": "assistant", "content": "我是您的專屬廚師"}]
HISTORY_LEN = len(HISTORY)


class HomePage(object):

    def __init__(self) -> None:

        if 'default_start_messages' not in st.session_state.keys():
            st.session_state.default_start_messages = "您好！我是您的生涯輔導師，很高興能幫助您一起探討和規劃您的生涯目標。請告訴我您的目前情況或挑戰，我會根據您的需求提供建議。"

        if 'history' not in st.session_state.keys():
            st.session_state.history = list(HISTORY)

        if 'assistant' not in st.session_state.keys():
            st.session_state.assistant = Assistant(
                url="http://192.168.112.10:5004/api/v1/llama8b/chat",
                system_prompt=SYSTEM_PROMPT_DEFAULT)

        if 'icon_dict' not in st.session_state.keys():
            st.session_state.icon_dict = {
                "user": "icon/user.jpg",
                "assistant": "icon/頭像.jpg"}

    @staticmethod
    def _create_box_system_prompt() -> str:

        st.sidebar.image(
            image=st.session_state.icon_dict["assistant"],
            width=200)
        st.sidebar.write("Model Name:")
        st.sidebar.write("meta-llama/Meta-Llama-3-8B-Instruct")
        st.sidebar.divider()
        return st.sidebar.text_area(label="System Prompt:",
                                    placeholder=st.session_state.assistant.system_prompt)

    @staticmethod
    def _create_new_chat() -> None:
        _bool = st.sidebar.button("New chat")
        if _bool:
            st.session_state.history = list(HISTORY)

    @staticmethod
    def _update_system_prompt(system_prompt: str) -> None:
        _bool = st.sidebar.button("Update System Prompt")
        if _bool:
            st.session_state.assistant.system_prompt = system_prompt

    @staticmethod
    def _reset_system_prompt() -> None:
        _bool = st.sidebar.button("Reset System Prompt")
        if _bool:
            st.session_state.assistant.system_prompt = SYSTEM_PROMPT_DEFAULT

    def main(self):

        box_str = HomePage._create_box_system_prompt()
        HomePage._update_system_prompt(box_str)
        HomePage._reset_system_prompt()
        HomePage._create_new_chat()
        assistant_triger_bool = False

        # Display the prior chat messages
        with st.chat_message(
                name="assistant",
                avatar=st.session_state.icon_dict["assistant"]):
            st.write(st.session_state.default_start_messages)
        for message in st.session_state.history[HISTORY_LEN:]:
            with st.chat_message(
                    name=message["role"],
                    avatar=st.session_state.icon_dict[message["role"]]):
                st.write(message["content"])

        # container for chat history
        user_chat_input = st.chat_input(
            placeholder="Chat with AI",
            key="chat_input")

        # Prompt for user input and display
        if user_chat_input is not None:
            logger.info(f"[user]: {user_chat_input}")
            assistant_triger_bool = True
            with st.chat_message(
                    name="user",
                    avatar=st.session_state.icon_dict["user"]):
                st.write(user_chat_input)

        # If user input, generate a new response
        if assistant_triger_bool:
            with st.chat_message(
                    name="assistant",
                    avatar=st.session_state.icon_dict["assistant"]):
                with st.spinner('Waiting...'):
                    streaming_box = st.empty()
                    try:
                        Free_start = time.time()
                        final_response = st.session_state.assistant.chat(
                            user_chat_input,
                            st.session_state.history)
                        with streaming_box.container():
                            st.markdown(final_response)
                        logger.info("執行時間：%f 秒" % (time.time() - Free_start))
                    except Exception as e:
                        logger.debug(e)
                        final_response = "請重新輸入\n"
                        st.write(final_response)
            # save history
            st.session_state.history.append({"role": "user", "content": user_chat_input})
            st.session_state.history.append({"role": "assistant", "content": final_response})
            logger.info(f"[AI]: {final_response}")
            logger.info(f"[history]: {st.session_state.history}")
            st.divider()
            assistant_triger_bool = False


if __name__ == "__main__":
    page = HomePage()
    page.main()

