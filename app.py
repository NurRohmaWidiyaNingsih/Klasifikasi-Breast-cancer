from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Path untuk menyimpan model
MODEL_PATH = 'stacking_classifier.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

def train_and_save_model():
    # Mengambil dataset dari file CSV
    breast = pd.read_csv('breast_cancer_2.csv')

    # Pisahkan fitur dan label
    feature_columns = ["Clump_thickness", "Uniformity_of_cell_size", "Uniformity_of_cell_shape", "Marginal_adhesion",
                       "Single_epithelial_cell_size", "Bare_nuclei", "Bland_chromatin", "Normal_nucleoli", "Mitoses"]
    X = breast[feature_columns].values
    y = breast['Class'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

    # Inisialisasi classifier KNN dengan k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Inisialisasi Stacking Classifier
    estimators = [('knn', knn)]
    stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    # Latih model pada set pelatihan
    stacking_classifier.fit(X_train, y_train)

    # Simpan model dan label encoder ke file
    joblib.dump(stacking_classifier, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

# Memuat model dan label encoder
def load_model():
    stacking_classifier = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return stacking_classifier, label_encoder

@app.route('/', methods=['GET', 'POST'])
def predict():
    stacking_classifier, label_encoder = load_model()
    
    if request.method == 'POST':
        try:
            clump_thickness = float(request.form['clump_thickness'])
            cell_size = float(request.form['cell_size'])
            cell_shape = float(request.form['cell_shape'])
            marginal_adhesion = float(request.form['marginal_adhesion'])
            epithelial_size = float(request.form['epithelial_size'])
            bare_nuclei = float(request.form['bare_nuclei'])
            bland_chromatin = float(request.form['bland_chromatin'])
            normal_nucleoli = float(request.form['normal_nucleoli'])
            mitoses = float(request.form['mitoses'])
            
            new_data = [[clump_thickness, cell_size, cell_shape, marginal_adhesion, epithelial_size, 
                         bare_nuclei, bland_chromatin, normal_nucleoli, mitoses]]
            
            prediction = stacking_classifier.predict(new_data)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            # Mengubah prediksi menjadi kategori 2 (jinak) atau 4 (ganas) dan menambahkan label
            if predicted_label == 2:
                predicted_class = "2 (Jinak)"
            else:
                predicted_class = "4 (Ganas)"
            
            return render_template('index.html', predicted_class=predicted_class)
        
        except ValueError as e:
            error_message = str(e)
            return render_template('index.html', error_message=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Latih dan simpan model saat pertama kali dijalankan
    train_and_save_model()
    app.run(debug=True)
