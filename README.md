# Spam-SMS-Classifier

This project is a **machine learning application built with Streamlit** that classifies emails or text messages as **Spam** or **Ham (Not Spam)**. It uses two popular models — **Logistic Regression** and **Naive Bayes** — trained on the classic `spam.csv` dataset.

### 🔍 Features
- Interactive **Streamlit web app** for classifying user‑entered messages.  
- **TF‑IDF vectorization** to convert text into numerical features.  
- Two models implemented:
  - Logistic Regression  
  - Multinomial Naive Bayes  
- Displays **accuracy, classification reports, and confusion matrix heatmaps** for both models.  
- Side‑by‑side comparison of model performance.  

### 🛠️ Tech Stack
- **Python**  
- **Streamlit** (for the web interface)  
- **Scikit‑learn** (for ML models and metrics)  
- **Pandas** (for data handling)  
- **Matplotlib & Seaborn** (for visualizations)  

### 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-email-classifier.git
   cd spam-email-classifier
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided local URL in your browser to interact with the app.

### 📊 Example Output
- Enter a message like:  
  ```
  Congratulations! You’ve won a free prize. Click here to claim.
  ```
- The app will classify it as **Spam 🚨**.  
- Reports show accuracy and confusion matrices for both models.
