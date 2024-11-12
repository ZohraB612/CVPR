import streamlit as st
from src.visualisation import plot_pr_curve, plot_confusion_matrix
# ... other imports ...

def main():
    st.title("Image Retrieval System")
    
    # Your existing code for image upload and processing...
    
    if results_available:  # After processing query image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Precision-Recall Curve")
            pr_fig = plot_pr_curve(pr_data, config)
            st.pyplot(pr_fig)
        
        with col2:
            st.subheader("Confusion Matrix")
            cm_fig = plot_confusion_matrix(distances, query_class)
            st.pyplot(cm_fig)
        
        # Display metrics
        st.subheader("Retrieval Metrics")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Average Precision", f"{ap:.3f}")
        with col4:
            st.metric("Precision@10", f"{precision_at_10:.3f}")
        with col5:
            st.metric("Precision@20", f"{precision_at_20:.3f}")

if __name__ == "__main__":
    main() 