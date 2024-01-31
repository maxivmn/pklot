import streamlit as st


st.set_page_config(
    page_title="Group",
)
#st.markdown("Group")
st.sidebar.header("Group Members")


members = [
    {"name": "Margarita", "image": "../images/unkn.png"},
    {"name": "Markus Neubeck", "image": "../images/unkn.png"},
    {"name": "Maximiliian", "image": "../images/unkn.png"},
    {"name": "Thisal Weerasekara", "image": "../images/unkn.png"},
    ]

    
for i in range(0, len(members), 2):
        cols = st.columns(2)  # Create two columns
        for j in range(2):
            # Check if there is a member to display in this column
            if i + j < len(members):
                member = members[i + j]
                with cols[j]:
                    st.markdown(f"## {member['name']}")
                    st.image(member["image"], width= 200)
                    #st.write(f"**Role:** {member['role']}")