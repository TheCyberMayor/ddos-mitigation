import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========== STEP 1: Load and tag each dataset ==========
def load_dataset(filepath, attack_name):
    df = pd.read_csv(filepath)
    df['AttackType'] = attack_name
    return df

print("Loading datasets...")
df_mssql = load_dataset('mssql_ddos.csv', 'MSSQL')
df_ntp   = load_dataset('ntp_ddos.csv', 'NTP')
df_udp   = load_dataset('udp_ddos.csv', 'UDP')
df_syn   = load_dataset('syn_ddos.csv', 'SYN')

# ========== STEP 2: Combine datasets ==========
df = pd.concat([df_mssql, df_ntp, df_udp, df_syn], ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# ========== STEP 3: Clean and preprocess ==========
df.dropna(inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Try to find the protocol column
protocol_col = None
for col in ['Protocol', 'ProtocolName']:
    if col in df.columns:
        protocol_col = col
        break

# Show protocol distribution
if protocol_col:
    print("\nProtocol Distribution:")
    print(df[protocol_col].value_counts())
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=protocol_col, order=df[protocol_col].value_counts().index)
    plt.title("Protocol Distribution by Attack Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("protocol_distribution.png")
    print("Protocol distribution chart saved as protocol_distribution.png")

# ========== STEP 4: Encode target labels ==========
le = LabelEncoder()
df['AttackType'] = le.fit_transform(df['AttackType'])  # e.g., 0=SYN, 1=UDP, 2=NTP, etc.

# ========== STEP 5: Select numeric features ==========
drop_cols = ['AttackType', protocol_col, 'Flow ID', 'Timestamp', 'Source IP', 'Destination IP',
             'Source Port', 'Destination Port', 'Label', 'Label Name'] if protocol_col else ['AttackType']

drop_cols = [col for col in drop_cols if col in df.columns]
X_raw = df.drop(columns=drop_cols)
X = X_raw.select_dtypes(include='number')
y = df['AttackType']

# ========== STEP 6: Normalize and split ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# ========== STEP 7: Train classifier ==========
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== STEP 8: Evaluation ==========
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
