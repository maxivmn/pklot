import streamlit as st

st.set_page_config(
    page_title="Home",layout='wide',
)

st.image('./images/ParkAI_9_transp.png',use_column_width=True)

st.header('Ai based parking detection :car:', divider='rainbow')

st.subheader('Problem Description')

st.write('In urban environments, the increasing number of vehicles poses a significant \
        challenge in managing parking spaces efficiently. Traditional methods of \
        monitoring parking availability often prove to be inefficient, leading to \
        wasted time and fuel consumption as drivers search for suitable parking \
        spots. By addressing these challenges and implementing the proposed solutions, \
        the Smart Car Parking Detection System aims to optimize parking space utilization,\
        reduce traffic congestion, and enhance the overall efficiency of urban mobility.')

st.write('To address this issue, a Smart Car Parking Detection System is proposed to enhance the management and utilization of parking spaces.')


st.subheader("Key Challenges :")
st.write("1. Limited Parking Spaces: \
         With the growing number of vehicles on the roads, parking spaces are becoming increasingly scarce. Efficient detection and management of available parking spaces are crucial to optimize utilization.")
st.write("2. Traffic Congestion: \
         The inability to quickly find an available parking spot contributes to traffic congestion as drivers circle around in search of parking. This not only wastes time but also increases carbon emissions.")
st.write("3. Inadequate Information: \
         Lack of real-time information about parking availability makes it challenging for drivers to plan their routes efficiently. Providing accurate and up-to-date information is essential for better traffic flow.")
st.write("4. Unauthorized Parking: \
         Illegally parked vehicles or those occupying spaces for extended periods contribute to inefficient use of parking facilities. A system is needed to monitor and manage these instances.")
st.write("5. Integration with Navigation Systems: \
         The system should seamlessly integrate with navigation applications to provide users with real-time updates on available parking spaces along their routes.")