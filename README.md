# AI-RAN Simulator (Customized Version)

This repository is a **customized version** of the [AI-RAN Simulator](https://github.com/ntutangyun/ai-ran-sim), originally developed by [ntutangyun](https://github.com/ntutangyun).  
It has been **modified and extended** to support additional functionalities such as **custom trace generation**, **traffic modeling**, and other simulation enhancements for advanced O-RAN research.

## 🏗️ Project Structure

```
ai-ran-sim/
├── backend/   # Modified Python simulation engine (core logic and custom modules)
├── frontend/  # Web interface for visualization and control
```

## 🚀 Getting Started

### Backend

```bash
pip install -r backend/requirements.txt
cd backend
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

## 🔧 Original Source

The original simulator is available here:  
👉 [https://github.com/ntutangyun/ai-ran-sim](https://github.com/ntutangyun/ai-ran-sim)

## 📘 Notes

This version includes several modifications to support:
- Custom **trace generation** and **aligned traffic modeling**
- Extended **state transitions** and **parameterized behavior** 
