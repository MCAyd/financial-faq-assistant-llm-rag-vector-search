import streamlit as st
import time
import uuid

from assistant import get_answer
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)

def print_log(message):
    print(message, flush=True)


def main():
    print_log("Starting the Financial Assistant application")
    st.title("Financial Assistant")

    # # Session state initialization
    if "count" not in st.session_state:
        st.session_state.count = 0
        print_log("Feedback count initialized to 0")

    # Course selection
    company_id = st.selectbox(
        "Select a company quote:",
        ["NVDA", "AAPL", "TSLA", "LULU", "PG", "COST", "ABNB", "MSFT", "BRK-A", 
        "META", "AXP", "PTON", "SBUX", "NKE", "PLTR", "AMZN", "NFLX", "GOOGL", 
        "ABBV", "V", "GME", "AMC", "CRM", "LLY", "AVGO", "UNH", "JNJ", "HD", 
        "WMT", "AMD", "CVX", "BAC", "KO", "T", "AZO", "CAT", "SCHW", "CMG", 
        "CB", "CMCSA", "CVS", "DVA", "DAL", "DLTR", "EBAY", "EA", "ENPH", 
        "EFX", "ETSY", "FDX", "F", "GRMN", "GIS", "GM", "GILD", "GS", "HAS", 
        "HSY", "HPE", "HLT", "HPQ", "HUM", "IBM", "ICE", "INTU", "IRM", 
        "JPM", "KR", "LVS"
        ],
    )
    print_log(f"User selected company: {company_id}")

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        # Generate a new conversation ID for next question
        st.session_state.conversation_id = str(uuid.uuid4())

        print_log(f"User asked: '{user_input}'")
        with st.spinner("Processing..."):
            print_log(
                f"Getting answer from assistant using gpt-4o-mini model and"
            )
            start_time = time.time()
            answer_data = get_answer(user_input, company_id)
            end_time = time.time()
            print_log(f"Answer received in {end_time - start_time:.2f} seconds")
            st.success("Completed!")
            st.write(answer_data["answer"])

            # Display monitoring information
            st.write(f"Response time: {answer_data['response_time']:.2f} seconds")
            st.write(f"Relevance: {answer_data['relevance']}")
            st.write(f"Model used: GPT-4o-mini")
            st.write(f"Search type used: Hybrid search with question-context vector")
            st.write(f"Total tokens: {answer_data['total_tokens']}")
            if answer_data["openai_cost"] > 0:
                st.write(f"OpenAI cost: ${answer_data['openai_cost']:.4f}")

            # Save conversation to database
            print_log("Saving conversation to database")
            save_conversation(
                st.session_state.conversation_id, user_input, answer_data, company_id
            )
            print_log("Conversation saved successfully")

    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            st.session_state.count += 1
            print_log(
                f"Positive feedback received. New count: {st.session_state.count}"
            )
            save_feedback(st.session_state.conversation_id, 1)
            print_log("Positive feedback saved to database")
    with col2:
        if st.button("-1"):
            st.session_state.count -= 1
            print_log(
                f"Negative feedback received. New count: {st.session_state.count}"
            )
            save_feedback(st.session_state.conversation_id, -1)
            print_log("Negative feedback saved to database")

    st.write(f"Current count: {st.session_state.count}")

    # Display recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox(
        "Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    )
    recent_conversations = get_recent_conversations(
        limit=5, relevance=relevance_filter if relevance_filter != "All" else None
    )
    for conv in recent_conversations:
        st.write(f"Q: {conv['question']}")
        st.write(f"A: {conv['answer']}")
        st.write(f"Relevance: {conv['relevance']}")
        st.write(f"Model: GPT-4o-mini")
        st.write("---")

    # Display feedback stats
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.write(f"Thumbs up: {feedback_stats['thumbs_up']}")
    st.write(f"Thumbs down: {feedback_stats['thumbs_down']}")


print_log("Streamlit app loop completed")


if __name__ == "__main__":
    print_log("Course Assistant application started")
    main()
