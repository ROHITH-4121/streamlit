import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# Page configuration
st.set_page_config(page_title="To-Do List", page_icon="✅", layout="wide")

# Title
st.title("✅ Smart To-Do List Manager")
st.markdown("### Organize your tasks and boost productivity")
st.markdown("---")

# Initialize session state
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

# Sidebar
st.sidebar.header("📌 Quick Stats")
if st.session_state.tasks:
    total_tasks = len(st.session_state.tasks)
    completed_tasks = len([t for t in st.session_state.tasks if t['status'] == 'Completed'])
    pending_tasks = total_tasks - completed_tasks
    
    st.sidebar.metric("Total Tasks", total_tasks)
    st.sidebar.metric("Completed", completed_tasks)
    st.sidebar.metric("Pending", pending_tasks)
    
    if total_tasks > 0:
        completion_rate = (completed_tasks / total_tasks) * 100
        st.sidebar.metric("Completion Rate", f"{completion_rate:.1f}%")
else:
    st.sidebar.info("No tasks yet. Start adding tasks!")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Priority Levels")
st.sidebar.markdown("""
- 🔴 **High**: Urgent tasks
- 🟡 **Medium**: Important tasks
- 🟢 **Low**: Can wait
""")

# Main tabs
tab1, tab2, tab3 = st.tabs(["➕ Add Task", "📋 My Tasks", "📊 Analytics"])

# Tab 1: Add Task
with tab1:
    st.header("Create New Task")
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_title = st.text_input("Task Title*", placeholder="e.g., Complete project report")
        task_description = st.text_area("Description", height=100,
                                       placeholder="Add details about the task...")
        task_priority = st.selectbox("Priority", ["High", "Medium", "Low"])
    
    with col2:
        task_category = st.selectbox("Category", 
                                    ["Work", "Personal", "Study", "Health", "Shopping", "Others"])
        task_due_date = st.date_input("Due Date", min_value=date.today())
        task_tags = st.text_input("Tags (comma-separated)", 
                                 placeholder="e.g., urgent, meeting, deadline")
    
    if st.button("➕ Add Task", type="primary", use_container_width=True):
        if task_title:
            new_task = {
                'id': len(st.session_state.tasks) + 1,
                'title': task_title,
                'description': task_description,
                'priority': task_priority,
                'category': task_category,
                'due_date': task_due_date.strftime('%Y-%m-%d'),
                'tags': task_tags,
                'status': 'Pending',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'completed_at': None
            }
            st.session_state.tasks.append(new_task)
            st.success("✅ Task added successfully!")
            st.balloons()
        else:
            st.error("Please enter a task title!")

# Tab 2: My Tasks
with tab2:
    st.header("All Tasks")
    
    if st.session_state.tasks:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_status = st.selectbox("Status", ["All", "Pending", "Completed"])
        with col2:
            categories = ["All"] + list(set([t['category'] for t in st.session_state.tasks]))
            filter_category = st.selectbox("Category", categories)
        with col3:
            priorities = ["All"] + list(set([t['priority'] for t in st.session_state.tasks]))
            filter_priority = st.selectbox("Priority", priorities)
        with col4:
            sort_by = st.selectbox("Sort by", ["Due Date", "Priority", "Created Date"])
        
        # Apply filters
        filtered_tasks = st.session_state.tasks.copy()
        
        if filter_status != "All":
            filtered_tasks = [t for t in filtered_tasks if t['status'] == filter_status]
        if filter_category != "All":
            filtered_tasks = [t for t in filtered_tasks if t['category'] == filter_category]
        if filter_priority != "All":
            filtered_tasks = [t for t in filtered_tasks if t['priority'] == filter_priority]
        
        # Sort tasks
        if sort_by == "Due Date":
            filtered_tasks = sorted(filtered_tasks, key=lambda x: x['due_date'])
        elif sort_by == "Priority":
            priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
            filtered_tasks = sorted(filtered_tasks, key=lambda x: priority_order[x['priority']])
        else:
            filtered_tasks = sorted(filtered_tasks, key=lambda x: x['created_at'], reverse=True)
        
        st.markdown(f"**Showing {len(filtered_tasks)} task(s)**")
        st.markdown("---")
        
        # Display tasks
        for idx, task in enumerate(filtered_tasks):
            # Priority emoji
            priority_emoji = "🔴" if task['priority'] == "High" else "🟡" if task['priority'] == "Medium" else "🟢"
            
            # Status color
            if task['status'] == 'Completed':
                status_color = "green"
            else:
                due_date = datetime.strptime(task['due_date'], '%Y-%m-%d').date()
                if due_date < date.today():
                    status_color = "red"
                elif due_date == date.today():
                    status_color = "orange"
                else:
                    status_color = "blue"
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"### {priority_emoji} {task['title']}")
                    st.markdown(f"**Category:** {task['category']} | **Due:** {task['due_date']} | **Status:** :{status_color}[{task['status']}]")
                    if task['description']:
                        st.markdown(f"*{task['description']}*")
                    if task['tags']:
                        tags = task['tags'].split(',')
                        tag_str = ' '.join([f"`{tag.strip()}`" for tag in tags])
                        st.markdown(f"🏷️ {tag_str}")
                
                with col2:
                    # Mark as complete/incomplete
                    if task['status'] == 'Pending':
                        if st.button("✅ Complete", key=f"complete_{task['id']}"):
                            for t in st.session_state.tasks:
                                if t['id'] == task['id']:
                                    t['status'] = 'Completed'
                                    t['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            st.success("Task completed!")
                            st.rerun()
                    else:
                        if st.button("↩️ Reopen", key=f"reopen_{task['id']}"):
                            for t in st.session_state.tasks:
                                if t['id'] == task['id']:
                                    t['status'] = 'Pending'
                                    t['completed_at'] = None
                            st.success("Task reopened!")
                            st.rerun()
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{task['id']}"):
                        st.session_state.tasks = [t for t in st.session_state.tasks if t['id'] != task['id']]
                        st.success("Task deleted!")
                        st.rerun()
                
                st.markdown("---")
    else:
        st.info("📝 No tasks yet. Add your first task to get started!")

# Tab 3: Analytics
with tab3:
    st.header("Task Analytics")
    
    if st.session_state.tasks:
        df = pd.DataFrame(st.session_state.tasks)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total = len(df)
        completed = len(df[df['status'] == 'Completed'])
        pending = total - completed
        overdue = len(df[(df['status'] == 'Pending') & (pd.to_datetime(df['due_date']).dt.date < date.today())])
        
        with col1:
            st.metric("Total Tasks", total)
        with col2:
            st.metric("Completed", completed, delta=f"{(completed/total)*100:.0f}%" if total > 0 else "0%")
        with col3:
            st.metric("Pending", pending)
        with col4:
            st.metric("Overdue", overdue, delta="⚠️" if overdue > 0 else "✅")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            st.subheader("📊 Task Status")
            status_counts = df['status'].value_counts()
            fig_status = px.pie(values=status_counts.values, 
                              names=status_counts.index,
                              color=status_counts.index,
                              color_discrete_map={'Completed':'#2ecc71', 'Pending':'#e74c3c'})
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            # Priority distribution
            st.subheader("🎯 Priority Distribution")
            priority_counts = df['priority'].value_counts()
            fig_priority = px.pie(values=priority_counts.values,
                                names=priority_counts.index,
                                color=priority_counts.index,
                                color_discrete_map={'High':'#e74c3c', 'Medium':'#f39c12', 'Low':'#2ecc71'})
            st.plotly_chart(fig_priority, use_container_width=True)
        
        # Category breakdown
        st.subheader("📁 Tasks by Category")
        category_counts = df['category'].value_counts()
        fig_category = px.bar(x=category_counts.index, 
                            y=category_counts.values,
                            labels={'x': 'Category', 'y': 'Number of Tasks'},
                            color=category_counts.values,
                            color_continuous_scale='Blues')
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Upcoming tasks
        st.subheader("📅 Upcoming Tasks (Next 7 Days)")
        df['due_date_dt'] = pd.to_datetime(df['due_date'])
        upcoming = df[(df['status'] == 'Pending') & 
                     (df['due_date_dt'] >= pd.Timestamp(date.today())) &
                     (df['due_date_dt'] <= pd.Timestamp(date.today()) + pd.Timedelta(days=7))]
        
        if len(upcoming) > 0:
            upcoming_display = upcoming[['title', 'category', 'priority', 'due_date']].copy()
            upcoming_display = upcoming_display.sort_values('due_date')
            st.dataframe(upcoming_display, use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming tasks in the next 7 days!")
        
        # Export data
        st.markdown("---")
        st.subheader("💾 Export Tasks")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"tasks_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Clear completed tasks
        st.markdown("---")
        st.subheader("🗑️ Clear Completed Tasks")
        completed_count = len(df[df['status'] == 'Completed'])
        
        if completed_count > 0:
            if st.button(f"🗑️ Clear {completed_count} Completed Task(s)", type="secondary"):
                st.session_state.tasks = [t for t in st.session_state.tasks if t['status'] != 'Completed']
                st.success("Completed tasks cleared!")
                st.rerun()
        else:
            st.info("No completed tasks to clear!")
        
    else:
        st.info("📊 No data available for analytics. Add some tasks first!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Stay productive! 🚀")
