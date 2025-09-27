# 📈 StockChat: Intelligent Stock Market Analysis Platform

StockChat is a powerful, interactive web application designed for stock market enthusiasts, providing real-time data analysis, portfolio tracking, and intelligent financial insights. It combines a user-friendly dashboard with advanced backend technologies, including a graph database for market relationship analysis and an AI-powered FAQ assistant.


<img width="940" height="475" alt="image" src="https://github.com/user-attachments/assets/fd868540-98ac-46fc-8efe-5aa7301cb476" />

## ✨ Key Features

This platform is built with several powerful features to provide a seamless user experience:

#### 1. 📊 Interactive Stock Dashboard
* **Real-time Data:** Fetches and displays live stock data from the Yahoo Finance API.
* **Advanced Charting:** View interactive Candlestick charts with key technical indicators like Moving Averages (MA20, MA50) and trading volume.
* **Key Metrics:** Instantly see vital statistics like current price, 52-week high/low, market cap, and daily change.
* **Market Overview:** Get a quick glance at the performance of major global indices like NIFTY 50, SENSEX, S&P 500, and NASDAQ.


<img width="940" height="481" alt="image" src="https://github.com/user-attachments/assets/44cd7cbf-afb2-4027-9066-dad908bb11ac" />
<img width="940" height="439" alt="image" src="https://github.com/user-attachments/assets/8a19608f-9baa-42cb-b869-de9b9a8390d3" />

---

#### 2. 👤 User Authentication & Personalized Watchlist (MySQL)
* **Secure User Accounts:** Users can sign up and log in to a secure account, with all user data managed through a **MySQL** database.
* **Custom Watchlist:** Add or remove stocks from a personalized watchlist to track investments that matter most to you.
* **Persistent Storage:** Your watchlist is saved to your account, so you can access it anytime you log in.

<img width="695" height="1454" alt="image" src="https://github.com/user-attachments/assets/a1151974-e76c-4e21-a240-0debe6779e86" />

---

#### 3. 🤖 Intelligent FAQ Assistant (RAG)
* **Conversational Q&A:** Ask complex questions about finance and the stock market in natural language.
* **RAG Architecture:** The assistant is powered by a **Retrieval-Augmented Generation (RAG)** model. It uses the `Sentence-Transformers` library to generate embeddings for user queries and retrieve the most relevant answers from a comprehensive financial knowledge base.
* **High Accuracy:** This approach ensures that answers are not just generated but grounded in factual, pre-defined data, providing reliable and contextually accurate information.

<img width="940" height="644" alt="image" src="https://github.com/user-attachments/assets/5cd54905-1990-467f-b7ce-2e6a4e8cc448" />

---

#### 4. 🔗 Advanced Market Intelligence (Neo4j)
* **Graph-Based Analytics:** The application uses a **Neo4j graph database** to model and query complex relationships between stocks, sectors, and portfolios.
* **Stock Correlation Analysis:** Discover how different stocks in the market move in relation to each other. The backend calculates Pearson correlation coefficients and stores these relationships in Neo4j for fast retrieval.
* **Portfolio Risk Assessment:** For users with a watchlist, the system analyzes portfolio concentration by sector, highlighting potential risks from over-diversification.
* **Diversification Suggestions:** Based on your current portfolio, the system queries the Neo4j graph to suggest investment opportunities in sectors where you have low exposure.

<img width="940" height="331" alt="image" src="https://github.com/user-attachments/assets/a2fb93ac-eab5-408a-903f-15e86b87483c" />


## 🛠️ Tech Stack

* **Frontend:** **Streamlit** for the interactive web interface, **Plotly** for data visualizations.
* **Backend:** **Python**
* **Data Retrieval:** **yfinance** library for fetching stock market data.
* **Databases:**
    * **MySQL:** For user authentication, session management, and storing personalized watchlists.
    * **Neo4j:** For storing and querying graph-based market data like stock correlations and sector relationships.
* **AI & Machine Learning:**
    * **Sentence-Transformers:** For creating text embeddings for the RAG-based FAQ system.
    * **Scikit-learn & SciPy:** For calculating cosine similarity and Pearson correlations.
* **Utilities:** **Pandas**, **NumPy** for data manipulation, **FuzzyWuzzy** for smart search suggestions.

