import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



def main():
    st.title('Common ML Dataset Explorer')
    st.subheader('Simple Data Science Explorer with Streamlit')
    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:60px;"> Streamlit is Awesome</p></div>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)
    st.subheader('Read and explorer Dataset')
    def file_selector(folder_path = '.\datasets'):
        filenames = os.listdir(folder_path)
        selected_file = st.selectbox('Select a file',filenames)
        return os.path.join(folder_path,selected_file)
    file_name = file_selector()
    st.info('You selected {}'.format(file_name))
    
    #read dataset
    df = pd.read_csv(file_name)
    
    #show dataset
    if st.checkbox('Show Dataset'):
        
        number = st.number_input('Number of rows to view',5,10)
        status = st.radio('Head/Tail',['Head','Tail'])
        if status =='Head':
            st.dataframe(df.head(number))
        else :
            st.dataframe(df.tail(number)) 
               
    #show cols
    if st.button('Column Name'):
        st.write(df.columns)
    #show shape
    if st.button('Shape of dataset'):
        st.write('Row : {} , Columns : {}'.format(df.shape[0],df.shape[1]))
    
    
    
    #select columns 
    if st.checkbox('Select columns to show'):
        all_cols = df.columns.to_list()
        select_cols = st.multiselect('Select',all_cols)
        df_new = df[select_cols]
        st.dataframe(df_new)
    
    # show values count
    if st.button('Values Counts'):
        st.text('Value Count by Target/Class')
        st.write(df.iloc[:,-1].value_counts())    
    #show datatype
    
    if st.button('Show Data Type'):
        st.write(df.dtypes)
    # show summary
    if st.checkbox('Summary'):
        st.write(df.describe().T)
    ## plot and visualization
    num_cols = df.columns[df.dtypes !='object']
    st.subheader('Pre Processing Data')
    st.write('Outlier')
    if st.checkbox('Boxplot for outlier'):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        for i, col in enumerate(num_cols[:6]):  # Giới hạn 6 cột đầu tiên
            row = i // 3  # Xác định dòng
            col_idx = i % 3  # Xác định cột
            sns.boxplot(data=df, y=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Boxplot of {col}', fontsize=14)
            axes[row, col_idx].tick_params(axis='x', labelsize=12)
            axes[row, col_idx].tick_params(axis='y', labelsize=12)

        # Hiển thị trong Streamlit
        st.pyplot(fig)
    st.write('Distributrion data')
    if st.checkbox('Distribution data'):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        for i, col in enumerate(num_cols[:6]):  # Giới hạn 6 cột đầu tiên
            row = i // 3  # Xác định dòng
            col_idx = i % 3  # Xác định cột
            sns.kdeplot(data=df, y=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(f'Kdeplot of {col}', fontsize=14)
            axes[row, col_idx].tick_params(axis='x', labelsize=12)
            axes[row, col_idx].tick_params(axis='y', labelsize=12)

        # Hiển thị trong Streamlit
        st.pyplot(fig)
    
    st.write('Balance data')
    if st.checkbox('Balance data'):
        # Đếm số lượng giá trị trong 'target'
        value_counts = df.iloc[:,-1].value_counts()

        # Sử dụng seaborn để chọn palette màu
        colors = sns.color_palette('Blues_d', len(value_counts))

        # Vẽ biểu đồ tròn
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)

        # Đảm bảo biểu đồ có hình tròn
        ax.axis('equal')

        # Thêm tiêu đề
        ax.set_title('Frequency Distribution of Target Categories', fontsize=16)

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)
    #corr
    if st.checkbox('Correlation plot'):
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[num_cols].corr()  # Tính toán ma trận tương quan
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)  # Vẽ heatmap
        st.pyplot(fig)
    if st.button('Thanks'):
        st.balloons()    
    
    #seaborn
    #pie char
    # customizable plot
    
if __name__  =='__main__':
    main()  
