# 🚦 Gridlock 2.0: AI Traffic Command Center

> Predict. Plan. Prevent.
>
> An AI-powered traffic operations platform that forecasts congestion from historical Bengaluru Traffic Police incident data and generates actionable response plans including manpower deployment, barricading strategies, diversion recommendations, and emergency response prioritization.

---

## 🏆 Flipkart Gridlock Hackathon 2.0

Gridlock 2.0 addresses the challenge of **Event-Driven Congestion Management** by combining:

- Historical Astram incident intelligence
- Machine Learning forecasting
- Explainable AI (XAI)
- Resource optimization
- Diversion planning
- Sustainability analytics
- Real-time traffic operations dashboards

The platform transforms traffic management from **reactive response** to **proactive decision-making**.

---

# 🚨 Problem Statement

Urban traffic incidents such as:

- Vehicle breakdowns
- Road closures
- Infrastructure failures
- Public events
- Unexpected congestion surges
- Emergency vehicle movement

can rapidly trigger corridor-wide gridlock.

Current traffic management is often reactive, leading to:

- Delayed officer deployment
- Increased congestion
- Fuel wastage
- Higher CO₂ emissions
- Slower emergency response

Gridlock 2.0 predicts congestion before it escalates and recommends optimal mitigation strategies.

---

# 🧠 Solution Architecture

```text
Historical Astram Data (8173 Events)
                 │
                 ▼
     Machine Learning Forecasting
                 │
                 ▼
      Traffic Impact Prediction
                 │
                 ▼
      Resource Allocation Engine
                 │
                 ▼
     Diversion Recommendation System
                 │
                 ▼
        Explainable AI Copilot
                 │
                 ▼
      AI Traffic Command Center
```

---

# ✨ Key Features

## 📊 Historical Traffic Intelligence

Built using:

- 8,173 Bengaluru Astram incident records
- Planned and unplanned traffic events
- Corridor-specific traffic patterns
- Historical operational outcomes

---

## 🤖 AI-Powered Congestion Forecasting

Predicts:

- Traffic impact score
- Congestion severity
- Queue formation risk
- Resource requirements
- Clearance estimates

using leakage-free machine learning models.

---

## 🚓 Smart Resource Allocation

Automatically recommends:

- Officer deployment
- Barricade allocation
- Tow units
- Response priorities
- Operational action plans

based on predicted traffic impact.

---

## 🛣️ AI Diversion Planning

Generates:

- Diversion recommendations
- Alternate route strategies
- Traffic redirection levels
- Emergency corridor prioritization

to minimize corridor disruption.

---

## 🚑 Emergency Vehicle Priority Mode

Special operational mode for:

- Ambulances
- Fire services
- Police response vehicles

Provides priority handling recommendations and accelerated response planning.

---

## 🔍 Explainable AI (XAI) Copilot

Every recommendation is explainable.

Example:

> Similar incidents involving road closures and congestion historically resulted in significant queue buildup. Recommended deployment: 8 officers and 4 barricades.

This increases trust and transparency for operational decision-makers.

---

# 🖥️ Platform Modules

## 🏠 AI Traffic Command Center

Real-time operational overview featuring:

- Total events analysed
- Response time saved
- CO₂ prevented
- Fuel waste avoided
- Average speed prediction
- Gridlock queue estimation

---

## 📊 Historical Analytics Dashboard

Historical intelligence and model evaluation:

- Event distribution
- Corridor analysis
- Cause breakdown
- Forecast metrics
- Feature importance
- Impact distribution

---

## 🚦 Live Operations Console

Real-time incident management interface featuring:

- Corridor selection
- Zone selection
- Event classification
- Priority management
- Emergency vehicle mode
- AI forecasting
- Resource recommendations
- Diversion planning
- Live telemetry monitoring

---

# 📈 Operational Impact

Computed using 8,173 historical Bengaluru traffic incidents.

| Metric | Value |
|----------|----------|
| Historical Events Analysed | 8,173 |
| Average Speed | 39.2 km/h |
| Average Queue per Event | 137 Vehicles |
| Median CO₂ Impact | 44.7 kg |
| Median Fuel Impact | ₹1,534 |
| Estimated Response Time Saved | 7.2 min/event |

---

# 🌱 Sustainability Impact

Gridlock 2.0 evaluates both operational and environmental outcomes.

### Telemetry Calculations

| Metric | Formula |
|----------|----------|
| Average Speed | Corridor Free Flow × Impact Reduction |
| Queue Length | Traffic Flow × Delay × Congestion Factor |
| Idle CO₂ | Queue × Idle Time × 0.0067 kg/veh/min |
| Fuel Cost | CO₂ ÷ 2.68 kg/L × ₹92/L |

Benefits:

- Reduced idle emissions
- Lower fuel wastage
- Faster congestion recovery
- Improved corridor efficiency

---

# 🏗️ Machine Learning Pipeline

## Impact Forecast Model

Leakage-free forecasting pipeline trained using:

- Event Type
- Event Cause
- Corridor
- Road Closure Indicators
- Priority Level
- Vehicle Category

Excluded from training:

- duration_minutes
- status

These fields are unknown at forecast time and would introduce target leakage.

---

## Model Performance

### Impact Forecaster

```text
CV MAE:   0.112 ± 0.042
CV RMSE:  1.701 ± 0.736
CV R²:    0.253 ± 0.336
CV MAPE:  2.96% ± 1.15%
```

The previous R² of 0.825 was found to be inflated due to target leakage.

The current evaluation reflects realistic pre-event forecasting performance.

---

### Resource Allocation Model

```text
Diversion Accuracy: 99.8%
Manpower MAE:       ~0.003
Barricades MAE:     ~0.003
```

---

# 🎯 Demo Workflow

1. Open Live Operations Console
2. Select active corridor
3. Choose event type and cause
4. Set priority level
5. Enable emergency mode if required
6. Run AI Forecast
7. Review impact score
8. Review officer deployment recommendation
9. Review barricade requirements
10. Review diversion strategy
11. Monitor environmental and operational impact

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <repository-url>
cd gridlock-2.0
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

```bash
python run_train.py
```

This trains:

1. Impact Forecast Model
2. Resource Allocation Model

Model artifacts are saved in:

```text
models/
```

---

# ▶️ Run Application

```bash
python -m src.app
```

Open:

```text
http://127.0.0.1:5000
```

---

# 📁 Project Structure

```text
gridlock-2.0/
├── run_train.py
├── requirements.txt
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── telemetry.py
│   └── xai.py
├── models/
├── templates/
├── static/
└── README.md
```

---

# 🔮 Future Scope

- Live GPS integration
- CCTV analytics integration
- Bengaluru-wide corridor expansion
- Real-time retraining pipelines
- Smart signal coordination
- Emergency green-wave corridors
- City-scale digital twin simulation
- OpenStreetMap integration

---

# 🎖️ Bengaluru Traffic Police Alignment

Gridlock 2.0 directly supports:

✅ Event-driven congestion management

✅ Resource optimization

✅ Emergency response prioritization

✅ Diversion planning

✅ Sustainability tracking

✅ Data-driven traffic operations

---

#  Team

Developed for **Flipkart Gridlock Hackathon 2.0**

**Gridlock 2.0 – From Traffic Prediction to Traffic Command.**