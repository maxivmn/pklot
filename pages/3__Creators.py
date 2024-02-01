import streamlit as st


st.set_page_config(
    page_title="Group",layout='wide',
)
#st.markdown("Group")
#st.sidebar.header("Group Members")
st.image("./images/ParkAI_9_transp.png",use_column_width=True)

st.header('Creators :car:', divider='rainbow')

members = [
    {"name": "Markus", "image": "./images/4_Markus.jpg"},
    {"name": "Margarita", "image": "./images/4_Margarita.jpeg"},
    {"name": "Thisal", "image": "./images/4_Thisal.jpg"},
    {"name": "Maximiliian", "image": "./images/4_Maxi.jpeg"},
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