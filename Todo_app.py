import streamlit as st
import pandas as pd
import os

FILE_PATH = "tasks.csv"

if "tasks" not in st.session_state:
  if os.path.exists(FILE_PATH):
    st.session_state.tasks = pd.read_csv(FILE_PATH)
  else:
    st.session_state.tasks = pd.DataFrame(columns=["Task", "Status"])

tasks = st.session_state.tasks

st.title("📝 To-Do List Web App")
st.write("Add, complete, and manage your daily tasks.")

st.subheader("➕ Add a New Task")
new_task = st.text_input("Enter task")

if st.button("Add Task"):
  if new_task.strip() != "":
    new_row = pd.DataFrame({"Task": [new_task], "Status": ["Pending"]})
    tasks = pd.concat([tasks, new_row], ignore_index=True)
    st.session_state.tasks = tasks
    tasks.to_csv(FILE_PATH, index=False)  # Save to CSV
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
      done_clicked = st.button("✅ Done", key=f"done{i}")
    with col3:
      delete_clicked = st.button("❌ Delete", key=f"del{i}")

    if done_clicked:
      tasks.at[i, "Status"] = "Completed"
      st.session_state.tasks = tasks
      tasks.to_csv(FILE_PATH, index=False)
      st.experimental_rerun()

    if delete_clicked:
      tasks = tasks.drop(i).reset_index(drop=True)
      st.session_state.tasks = tasks
      tasks.to_csv(FILE_PATH, index=False)
      st.experimental_rerun()

st.subheader("✔️ Completed Tasks")
completed = tasks[tasks["Status"] == "Completed"]
if not completed.empty:
    st.dataframe(completed)
