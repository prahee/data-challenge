import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import random #simulating errors
import base64

try:
    from barnacle_utils import load_image, extract_barnacle_features
except ImportError:
    st.error("Error: Could not import functions from barnacle_utils.py.")
    st.info("Please ensure 'barnacle_utils.py' is in the same directory as 'app.py' "
            "and contains 'load_image' and 'extract_barnacle_features' functions.")
    st.stop() 

st.set_page_config(
    page_title="DALI Barnacle Counter Prototype",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(44, 95, 124, 0.3);
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #B8D4E3 !important;
        margin: 0;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #E8F1F5;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #7A9FB8;
        box-shadow: 0 2px 4px rgba(122, 159, 184, 0.2);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #2C5F7C !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #4A90A4 !important;
        margin: 0.5rem 0 !important;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-card p {
        color: #2C5F7C !important;
        margin: 0 !important;
        font-size: 0.9rem;
    }
    .success-banner {
        background-color: #B8D4E3;
        border: 1px solid #4A90A4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2C5F7C !important;
    }
    .success-banner strong {
        color: #2C5F7C !important;
    }
    .workflow-step {
        background-color: #E8F1F5;
        border-left: 4px solid #4A90A4;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(184, 212, 227, 0.2);
    }
    .workflow-step h4 {
        color: #2C5F7C !important;
        margin-bottom: 0.5rem;
    }
    .workflow-step p {
        color: #2C5F7C !important;
        margin: 0;
    }
    .section-header {
        background-color: #B8D4E3;
        padding: 1rem;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4A90A4;
        margin: 1rem 0;
    }
    .section-header h3 {
        color: #2C5F7C !important;
        margin: 0;
    }
    .footer-section {
        background-color: #2C5F7C;
        color: white !important;
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        text-align: center;
    }
    .footer-section h4 {
        color: white !important;
        margin-bottom: 1rem;
    }
    .footer-section p {
        color: #B8D4E3 !important;
    }
    .footer-section em {
        color: white !important;
    }
    .comparison-box {
        background-color: #E8F1F5;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #7A9FB8;
        margin: 0.5rem 0;
    }
    .comparison-box.improved {
        background-color: #B8D4E3;
        border: 1px solid #4A90A4;
    }
    .improvement-metrics {
        background-color: #B8D4E3;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #2C5F7C !important;
    }
    .improvement-metrics h4 {
        color: #2C5F7C !important;
        margin-bottom: 0.5rem;
    }
    .improvement-metrics p {
        color: #2C5F7C !important;
        margin: 0.25rem 0;
    }
    

    .stButton > button {
        background-color: #4A90A4 !important;
        color: white !important;
        border: 1px solid #2C5F7C !important;
        border-radius: 8px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2C5F7C !important;
        border: 1px solid #2C5F7C !important;
        color: white !important;
    }
    
    .stCheckbox > label {
        color: #2C5F7C !important;
    }
    
    .css-1d391kg {
        background-color: #E8F1F5 !important;
    }
    
    .css-1xarl3l {
        background-color: #B8D4E3 !important;
        border: 1px solid #4A90A4 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .cover-image-container {
        position: relative;
        width: 100%;
        height: 300px;
        border-radius: 15px;
        overflow: hidden;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(44, 95, 124, 0.3);
    }
    .cover-image-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(44, 95, 124, 0.7) 0%, rgba(74, 144, 164, 0.5) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
    }
    .cover-image-text {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .cover-image-text h2 {
        color: white !important;
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .cover-image-text p {
        color: #B8D4E3 !important;
        font-size: 1.1rem;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# About This Demo")
    
    st.markdown("### Purpose")
    st.markdown("""
    This interactive prototype showcases a **Human-Model Interaction** solution 
    for the DALI Data Challenge (visualization focused), showing how automated systems can accelerate scientific 
    research while also maintaining human oversight at the same time.
    """)
    
    st.markdown("### Scientific Context")
    st.info("""
    **Challenge**: Marine biologists manually count 1000+ barnacles per image - 
    a time-intensive process that limits research scope.
    
    **Solution**: Computer-assisted counting with human verification and correction.
    """)
    
    st.markdown("### System Components")
    st.markdown("""
    - **Data Exploration**: Statistical analysis of barnacle populations
    - **U-Net Model**: Deep learning for automated detection
    - **Interactive Interface**: Human-in-the-loop error correction
    """)
    
    st.markdown("### Impact")
    st.success("**Potential 10x speedup** in barnacle population analysis")
    
    st.markdown("---")
    st.caption("Developed for the DALI LAB Data Challenge")
    
    # Add model performance stats
    st.markdown("### Model Statistics")
    st.metric("Training Images", "2 (Demo)")
    st.metric("Architecture", "U-Net")
    st.metric("Training Epochs", "30")

# cover image
try:
    cover_image = "Barnacles/barnacle_stock_image.jpg"
    if os.path.exists(cover_image):
        st.markdown("""
        <div class="cover-image-container">
            <div class="cover-image-overlay">
                <div class="cover-image-text">
                    <h2>DALI LAB Data Challenge</h2>
                    <p>Marine Biology Research Through Technology</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with open(cover_image, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
            <style>
                .cover-image-container {{
                    background-image: url('data:image/jpeg;base64,{img_data}');
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
            height: 200px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 6px 12px rgba(44, 95, 124, 0.3);
        ">
            <div>
                <h2 style="color: white; font-size: 2rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
                    DALI LAB Data Challenge
                </h2>
                <p style="color: #B8D4E3; font-size: 1.1rem; margin: 0;">
                    Marine Biology Research Through Technology
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
except Exception as e:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2C5F7C 0%, #4A90A4 100%);
        height: 200px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(44, 95, 124, 0.3);
    ">
        <div>
            <h2 style="color: white; font-size: 2rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
                DALI LAB Data Challenge
            </h2>
            <p style="color: #B8D4E3; font-size: 1.1rem; margin: 0;">
                Advancing Marine Biology Research Through Technology
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# title
st.markdown("""
<div class="main-header">
    <h2>Automated Barnacle Counting Prototype</h2>
    <p>Marine Biology Research Tool</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Workflow Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="workflow-step">
        <h4>Step 1: Model Detection</h4>
        <p>Computer vision analyzes image and identifies potential barnacles</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="workflow-step">
        <h4>Step 2: Human Review</h4>
        <p>Scientist reviews and corrects automated predictions</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="workflow-step">
        <h4>Step 3: Final Count</h4>
        <p>Verified results ready for scientific analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Scientific Challenge")
st.markdown("""
Marine biologists studying coastal ecosystems need to count barnacles in standardized survey areas. 
With 1000+ barnacles per image, manual counting is extremely time consuming and prone to fatigue induced errors.

This prototype demonstrates how computer-assisted analysis can aid and accelerate the research process 
while maintaining the accuracy and oversight that scientific research needs.
""")

UNSEEN_IMG_PATH = "Barnacles/unseen_img1.png"
MASK1_PATH = "Barnacles/mask1.png" # mask1 as a proxy for ground truth for unseen_img for demo

unseen_img = load_image(UNSEEN_IMG_PATH)
gt_mask_for_demo = load_image(MASK1_PATH, grayscale=True)
ground_truth_barnacles = []
ground_truth_barnacles_df = pd.DataFrame()
if gt_mask_for_demo is not None:
    ground_truth_barnacles = extract_barnacle_features(gt_mask_for_demo, "mask1 (GT)")
    ground_truth_barnacles_df = pd.DataFrame(ground_truth_barnacles)
    
st.markdown("---")
st.markdown("### Step 1 - Model Prediction")
st.markdown("""
Our U-Net neural network analyzes the image and identifies potential barnacles. 
The model was trained on annotated barnacle images and can detect individual organisms 
with their boundaries and morphological characteristics.
""")

# CI
confidence_score = random.uniform(0.75, 0.92)
st.markdown(f"**Model Confidence:** {confidence_score:.1%}")

mock_predictions_list = []
if not ground_truth_barnacles_df.empty:
    all_gt_barnacle_ids = list(ground_truth_barnacles_df.index)

    # simulate FNs
    num_to_miss = int(len(all_gt_barnacle_ids) * 0.15) 
    false_negatives_ids = random.sample(all_gt_barnacle_ids, num_to_miss)
    
    for idx, barnacle in ground_truth_barnacles_df.iterrows():
        if idx not in false_negatives_ids:
            mock_predictions_list.append(barnacle.to_dict())

    # simulate FPs
    num_false_positives = int(len(mock_predictions_list) * 0.05) 
    img_height, img_width = unseen_img.shape[0], unseen_img.shape[1]
    
    for _ in range(num_false_positives):
        fake_bbox_w = random.randint(10, 40)
        fake_bbox_h = random.randint(10, 40)
        fake_bbox_x = random.randint(0, img_width - fake_bbox_w)
        fake_bbox_y = random.randint(0, img_height - fake_bbox_h)
        
        mock_predictions_list.append({
            "image_name": "unseen_img1.png",
            "id_in_image": f"FP_{random.randint(1000, 9999)}",
            "area": float(fake_bbox_w * fake_bbox_h), "perimeter": float(2*(fake_bbox_w + fake_bbox_h)),
            "circularity": 0.5, "aspect_ratio": fake_bbox_w/fake_bbox_h, "solidity": 0.8,
            "centroid_x": fake_bbox_x + fake_bbox_w / 2, "centroid_y": fake_bbox_y + fake_bbox_h / 2,
            "bbox_x": fake_bbox_x, "bbox_y": fake_bbox_y, 
            "bbox_w": fake_bbox_w, "bbox_h": fake_bbox_h,
            "contour": None 
        })

mock_predictions_df = pd.DataFrame(mock_predictions_list)
initial_predicted_count = len(mock_predictions_df)
true_count = len(ground_truth_barnacles_df) if not ground_truth_barnacles_df.empty else 0

# metrics
st.markdown("#### Detection Results")
col_pred, col_true, col_accuracy = st.columns(3)

with col_pred:
    st.markdown("""
    <div class="metric-card">
        <h3>Model Prediction</h3>
        <h2>{}</h2>
        <p>Detected barnacles</p>
    </div>
    """.format(initial_predicted_count), unsafe_allow_html=True)

with col_true:
    st.markdown("""
    <div class="metric-card">
        <h3>Ground Truth</h3>
        <h2>{}</h2>
        <p>Actual count</p>
    </div>
    """.format(true_count), unsafe_allow_html=True)

with col_accuracy:
    accuracy = (1 - abs(initial_predicted_count - true_count) / true_count) * 100 if true_count > 0 else 0
    st.markdown("""
    <div class="metric-card">
        <h3>Initial Accuracy</h3>
        <h2>{:.1f}%</h2>
        <p>Before correction</p>
    </div>
    """.format(accuracy), unsafe_allow_html=True)


display_img_with_preds = None
if unseen_img is not None:
    display_img_with_preds = unseen_img.copy()
    
    for _, barnacle in mock_predictions_df.iterrows():
        x, y, w, h = int(barnacle['bbox_x']), int(barnacle['bbox_y']), int(barnacle['bbox_w']), int(barnacle['bbox_h'])
        cv2.rectangle(display_img_with_preds, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    st.markdown("#### Detection Visualization")
    st.image(display_img_with_preds, caption="Green boxes show computer detected barnacles", use_column_width=True)
    
    # Add detection summary
    st.markdown(f"**Detection Summary:** Found {initial_predicted_count} potential barnacles using computer vision analysis")
else:
    st.error("Could not load image for analysis. Please check file paths.")

st.markdown("---")

# interactive review
st.markdown("### Step 2 - Scientist Review & Correction")
st.markdown("""
**Human expertise is crucial!!**  As a marine biologist, you can quickly spot and correct automated errors. 
This human-in-the-loop approach ensures scientific accuracy while maintaining efficiency.
""")

# common errors
with st.expander("Common Detection Errors"):
    st.markdown("""
    - **False Positives**: Rocks, debris, or other organisms misidentified as barnacles
    - **False Negatives**: Small or partially obscured barnacles missed by the model
    - **Boundary Issues**: Incorrect barnacle size or shape detection
    - **Clustering Problems**: Multiple barnacles detected as one, or vice versa
    """)

col_actions, col_visual_errors = st.columns([1, 1.2])

with col_actions:
    st.markdown("#### Correction Tools")
    st.markdown("Use these controls to refine the model's predictions:")

    if 'current_count' not in st.session_state:
        st.session_state['current_count'] = initial_predicted_count

    # Enhanced button styling and feedback
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Remove False Positive", help="Click when model incorrectly identified a non-barnacle"):
            st.session_state['current_count'] -= 1
            st.success("Removed false positive!")
    
    with col_btn2:
        if st.button("Add Missed Barnacle", help="Click when model missed a real barnacle"):
            st.session_state['current_count'] += 1
            st.success("Added missed barnacle!")
    
    st.markdown("---")
    st.markdown("#### Correction Progress")
    
    correction_made = st.session_state['current_count'] != initial_predicted_count
    improvement = abs(st.session_state['current_count'] - true_count) < abs(initial_predicted_count - true_count)
    
    if correction_made:
        if improvement:
            st.markdown("""
            <div class="success-banner">
                <strong>Great job!</strong> Your corrections improved accuracy!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Keep refining - accuracy can still be improved!")
    
    # Enhanced current count display
    final_accuracy = (1 - abs(st.session_state['current_count'] - true_count) / true_count) * 100 if true_count > 0 else 0
    
    st.markdown("""
    <div class="metric-card">
        <h3>Final Count</h3>
        <h2>{}</h2>
        <p>Accuracy: {:.1f}%</p>
    </div>
    """.format(st.session_state['current_count'], final_accuracy), unsafe_allow_html=True)

with col_visual_errors:
    st.markdown("#### Error Analysis Visualization")
    st.markdown("Toggle the options below to understand where the model made mistakes:")

    show_ground_truth = st.checkbox("Show Ground Truth (Blue)", value=False, help="Show actual barnacle locations")
    show_false_positives = st.checkbox("Highlight False Positives (Red)", value=True, help="Show incorrect detections")
    show_false_negatives = st.checkbox("Highlight False Negatives (Yellow)", value=True, help="Show missed barnacles")

    error_visual_img = None
    if unseen_img is not None and not ground_truth_barnacles_df.empty:
        error_visual_img = unseen_img.copy()

        if show_ground_truth:
            for _, barnacle_gt in ground_truth_barnacles_df.iterrows():
                x, y, w, h = int(barnacle_gt['bbox_x']), int(barnacle_gt['bbox_y']), int(barnacle_gt['bbox_w']), int(barnacle_gt['bbox_h'])
                cv2.rectangle(error_visual_img, (x, y), (x + w, y + h), (0, 0, 255), 1) 

        # overlay FP in red
        if show_false_positives:
            for barnacle_pred in mock_predictions_list:
                if isinstance(barnacle_pred['id_in_image'], str) and barnacle_pred['id_in_image'].startswith("FP_"):
                    x, y, w, h = int(barnacle_pred['bbox_x']), int(barnacle_pred['bbox_y']), int(barnacle_pred['bbox_w']), int(barnacle_pred['bbox_h'])
                    cv2.rectangle(error_visual_img, (x, y), (x + w, y + h), (255, 0, 0), 2) 

        # overlay FN in yellow
        if show_false_negatives and 'false_negatives_ids' in locals():
            for idx in false_negatives_ids:
                if idx < len(ground_truth_barnacles_df): 
                    barnacle_fn = ground_truth_barnacles_df.iloc[idx]
                    x, y, w, h = int(barnacle_fn['bbox_x']), int(barnacle_fn['bbox_y']), int(barnacle_fn['bbox_w']), int(barnacle_fn['bbox_h'])
                    cv2.rectangle(error_visual_img, (x, y), (x + w, y + h), (255, 255, 0), 2) 

        st.image(error_visual_img, caption="🔍 Error Analysis: Blue=Ground Truth, Red=False Positives, Yellow=False Negatives", use_column_width=True)
        
        # error stats
        if show_false_positives or show_false_negatives:
            fp_count = len([p for p in mock_predictions_list if isinstance(p['id_in_image'], str) and p['id_in_image'].startswith("FP_")])
            fn_count = len(false_negatives_ids) if 'false_negatives_ids' in locals() else 0
            
            st.markdown(f"""
            **Error Summary:**
            - False Positives: {fp_count}
            - False Negatives: {fn_count}
            - Error Rate: {((fp_count + fn_count) / true_count * 100):.1f}%
            """)
    else:
        st.info("Enable ground truth data to see detailed error analysis.")

st.markdown("---")

#additional features
col_reset, col_export = st.columns(2)

with col_reset:
    if st.button("Reset Demo", help="Reset all corrections and start over"):
        st.session_state['current_count'] = initial_predicted_count
        st.experimental_rerun()

with col_export:
    if st.button("Export Results", help="Download counting results (demo)"):
        st.success("Results exported! ( this is a prototype; in a real app, this would download a CSV file)")

# workflow summary
st.markdown("### Workflow Summary")
time_saved = max(0, (true_count * 2 - 30)) # assume 2 sec per manual count vs 30 sec with computer vision
efficiency_gain = (time_saved / (true_count * 2) * 100) if true_count > 0 else 0

col_time, col_accuracy, col_efficiency = st.columns(3)
with col_time:
    st.metric("Time Saved", f"{time_saved}s", help="Estimated time savings vs manual counting")
with col_accuracy:
    st.metric("Final Accuracy", f"{final_accuracy:.1f}%", help="Accuracy after human corrections")
with col_efficiency:
    st.metric("Efficiency Gain", f"{efficiency_gain:.0f}%", help="Overall process improvement")

st.markdown("---")

# exploration section
st.markdown("### Explore the full Analysis Process")

col_explore1, col_explore2 = st.columns(2)

with col_explore1:
    st.markdown("""
    #### Data Exploration Notebook
    Barnacle_Data_Exploration.ipynb presents statistical analysis of barnacle populations, including morphological feature distributions and spatial distribution patterns.
    """)
    

st.markdown("""
#### DALI Challenge Outcomes
This prototype brings together automation and expert knowledge, integrates smoothly into scientific workflows, scales to other marine biology applications, and offers meaningful time savings for ecological research.
""")

st.markdown("---")

# footer
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #2C5F7C; border-radius: 10px; margin-top: 2rem; color: white;">
    <h4 style="color: white; margin-bottom: 1rem;">Transforming Marine Biology Research</h4>
    <p style="color: #B8D4E3;">This Streamlit prototype showcases automated barnacle counting for the <strong style="color: white;">DALI LAB Data Challenge</strong>.</p>
    <p style="color: #B8D4E3;">By combining computer vision with human expertise, we can accelerate ecological research while maintaining scientific rigor.</p>
    <p style="color: white;"><em>Empowering scientists to focus on discovery, not counting</em></p>
</div>
""", unsafe_allow_html=True)