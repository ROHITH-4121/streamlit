import streamlit as st
import pandas as pd
import os

FILE_PATH = "tasks.csv"

if os.path.exists(FILE_PATH):
  tasks = pd.read_csv(FILE_PATH)
else:
  tasks = pd.DataFrame(columns=["Task", "Status"])

st.title("📝 To-Do List Web App")
st.write("Add, complete, and manage your daily tasks.")

st.subheader("➕ Add a New Task")
new_task = st.text_input("Enter task")

if st.button("Add Task"):
  if new_task.strip() != "":
    new_row = pd.DataFrame({"Task": [new_task], "Status": ["Pending"]})
    tasks = pd.concat([tasks, new_row], ignore_index=True)
    tasks.to_csv(FILE_PATH, index=False)
    st.success("✅ Task added successfully!")
  else:
    st.warning("⚠️ Please enter a valid task.")

st.subheader("📋 Your Tasks")
if tasks.empty:
  st.info("No tasks yet. Add one above.")
else:
  for i, row in tasks.iterrows():
    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
      st.write(row["Task"])
    with col2:
      if st.button("✅ Done", key=f"done{i}"):
        tasks.at[i, "Status"] = "Completed"
        tasks.to_csv(FILE_PATH, index=False)
        st.experimental_rerun()
    with col3:
      if st.button("❌ Delete", key=f"del{i}"):
        tasks = tasks.drop(i).reset_index(drop=True)
        tasks.to_csv(FILE_PATH, index=False)
        st.experimental_rerun()

st.subheader("✔️ Completed Tasks")
completed = tasks[tasks["Status"] == "Completed"]
if not completed.empty:
  st.dataframe(completed)
