import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
#from sklearn.metrics.pairwise import cosine_similarity
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from tueplots import bundles

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({'figure.dpi': 200})



def generate_subset():
    path = "C:/Honors Project/Social Network Analysis/Data/"
    df1 =pd.read_csv(path+"cleaned_student_term_course_section(20211021).csv",low_memory=False)
    df2 = pd.read_csv(path+"Student_Term_20220708.csv", encoding = "ISO-8859-1")

    # create a unique course identifier
    df1["term_code"], df1["course_code"], df1["course_section_num"] = df1["term_code"].astype(str), df1["course_code"].astype(str), df1["course_section_num"].astype(str)
    df1["course_id"] = df1["term_code"]+"_"+df1["course_code"]+"_"+df1["course_dept_code_and_num"]+"_"+df1["course_section_num"]
    d_courses = {course:i for i,course in enumerate(df1["course_id"].unique())}
    df1["course_id_int"] = [d_courses[i] for i in df1["course_id"]]

    # merge df1 and df2
    df2.rename(columns={"#mellon_id": "mellon_id" }, inplace = True)
    df1["term_code"] = df1["term_code"].astype(int)
    data = df1.merge(df2,how="left",on=["mellon_id","term_code","term_desc"])
    data["total"] = 1
    data = data.dropna(axis=1,how="all") # drop empty columns

    # Filter by major_school_name
    data = data.loc[(~data["major_school_name_1"].isin(['Summer Session','Unknown',' '])) & (~data["major_school_name_1"].isnull())]
    data = data.reset_index(drop=True)

    # Filter by freshmen
    subset_freshmen_ids = data.loc[(data.year_study=="Freshman") & ((data.term_code >= 201192) & (data.term_code <= 201692))]["mellon_id"].unique()
    data = data.loc[data.mellon_id.isin(subset_freshmen_ids)].reset_index(drop=True)
    data = data.loc[data.term_code >= 201192]

    # Filter by number of terms
    stud_num_terms = agg_to_plot_bar(data,"term_code",var=False)
    stud_complete = stud_num_terms[stud_num_terms.freq >= 12]["mellon_id"].to_list() # CHECK THIS!
    data = data.loc[data.mellon_id.isin(stud_complete)] 

    # Only passed courses
    data = data.loc[~(data["final_grade"].isnull() | data["final_grade"].isin(["F","I","IP"," ","UR","W","U","NP","NR"]) )] 
    return data


def drop_one_student_course(data):
    df = data.loc[:,["mellon_id","course_id"]].copy()
    df = df.groupby("course_id")['mellon_id'].apply(list).reset_index()
    df.columns = ["course_id","mellon_id_list"]
    df["course_size"] = df["mellon_id_list"].apply(len)
    one_stud_courses = df.loc[df["course_size"]>1,"course_id"].to_list()
    return data.loc[data["course_id"].isin(one_stud_courses)]

def agg_to_plot_bar(df,var_col,unit_col="mellon_id",var=True):
    df_agg = (df.loc[:,[unit_col,var_col]]
               .groupby([unit_col,var_col])
               .count()
               .reset_index()
               )
    
    df_agg["freq"] = 1
    if var:
        df_plot = (df_agg.groupby(var_col)
                .sum()
                .reset_index()
                .sort_values(var_col)
                )
        df_plot = df_plot.loc[:,[var_col,"freq"]]
    else:
        df_plot = (df_agg.groupby(unit_col)
                .sum()
                .reset_index()
                .sort_values(unit_col)
                )
        df_plot = df_plot.loc[:,[unit_col,"freq"]]
    #df_plot[var_col] = df_plot[var_col].astype(str)
    return df_plot

def plot_bar(df_plot,title,data_labels=False,as_str=True,rot=90):
    """
    df_plot (pd.DataFrame): 2 columns, column 0 - x axis, column 1 - y axis (frequency)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    if as_str:
        df_plot.iloc[:,0] = df_plot.iloc[:,0].astype(str)
    bar_container = ax.bar(df_plot.iloc[:,0],df_plot.iloc[:,1],width=0.8)
    #ax.set_xlabel(var_col)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', labelrotation = rot)
    ax.set_title(title)
    if data_labels:
        ax.bar_label(bar_container, fmt='{:,.0f}')
    plt.show()

def plot_bar_perc(df_plot,title,data_labels=False,as_str=True,rot=90):
    """
    df_plot (pd.DataFrame): 2 columns, column 0 - x axis, column 1 - y axis (frequency)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    if as_str:
        df_plot.iloc[:,0] = df_plot.iloc[:,0].astype(str)
    df_plot.iloc[:,1] = df_plot.iloc[:,1] / df_plot.iloc[:,1].sum()
    bar_container = ax.bar(df_plot.iloc[:,0],df_plot.iloc[:,1],width=0.8)
    #ax.set_xlabel(var_col)
    ax.set_ylabel("Share")
    ax.tick_params(axis='x', labelrotation = rot)
    ax.set_title(title)
    if data_labels:
        ax.bar_label(bar_container, fmt='{:,.0f}')
    plt.show()
    
def adj_thr(adj,thr=1):
    n_courses = adj.sum(axis=1)
    sim = (adj @ adj.T)/ n_courses # PERCENTAGE
    sim = csr_matrix(sim)
    sim.setdiag(0)
    sim = 1*(sim>=thr)
    sim = 1*((sim+sim.T)>=1)
    return sim.A

def stud_by_course_matrix(df,stud_colname,course_colname,values_colname):

    student_u = CategoricalDtype(sorted(df[stud_colname].unique()), ordered=True) 
    course_u = CategoricalDtype(sorted(df[course_colname].unique()), ordered=True) 

    row = df[stud_colname].astype(student_u).cat.codes
    col = df[course_colname].astype(course_u).cat.codes
    sparse_matrix = csr_matrix((df[values_colname], (row, col)), shape=(student_u.categories.size, course_u.categories.size))
    print(f"Average number of courses: {sparse_matrix.sum(axis=1).mean() }")
    print(f"Shape of Adjacency matrix: {sparse_matrix.shape}")
    return sparse_matrix, student_u.categories.to_list(), course_u.categories.to_list()

def find_cliques(adj,thr=1):
    """
    Finds cliques in the adjacency matrix and returns two vectors: cliques_idx and cliques_binary

    Input:
    - adj (scipy.sparse.csr_matrix): a NxM matrix, where rows correspond to an unique student (N students) and columns - to courses (M unique courses)
    - thr (int): the percentage of courses taken by a group of students together that makes them a clique. 
                    By default thr = 1, which means that students form a clique only when all of the courses thay took intersect.
    Returns:
    cliques_idx (N,) - a vector of indices corresponding to a unique cliques
    cliques_binary (N,) - a vector of ones and zeros, one indicates that a student is in a clique.
    """
    n_courses = adj.sum(axis=1)
    sim = (adj @ adj.T)/ n_courses # PERCENTAGE
    sim = csr_matrix(sim)
    sim.setdiag(0)
    sim = 1*(sim>=thr)
    sim = 1*((sim+sim.T)>=1)
    #sim = cosine_similarity(adj) # similar to percentage but adjust to different vector lengths, if lengths are the same, similarity=1
    #np.fill_diagonal(sim,0)
    clique_size = sim.sum(axis=1).A.flatten()
    return clique_size

def clique_to_df(adj,stud_ids,ids_in_df,thr=1):
    clique_size = find_cliques(adj,thr=thr)
    print(f"Total number of students in cliques: {sum(clique_size>0)}")
    print(np.sort(clique_size)[-20:])

    d_studid_cliq = dict(zip(stud_ids,clique_size))
    clique_studids = np.array(stud_ids)[clique_size>0]
    return [d_studid_cliq[stud] for stud in ids_in_df], sorted(list(clique_studids ))

def clique_thr_test(df,stud_colname,course_colname,values_colname,thr=1,size=False):
    student_u = CategoricalDtype(sorted(df[stud_colname].unique()), ordered=True) 
    course_u = CategoricalDtype(sorted(df[course_colname].unique()), ordered=True) 

    row = df[stud_colname].astype(student_u).cat.codes
    col = df[course_colname].astype(course_u).cat.codes
    adj = csr_matrix((df[values_colname], (row, col)), shape=(student_u.categories.size, course_u.categories.size))

    n_courses = adj.sum(axis=1)
    sim = (adj @ adj.T)/ n_courses # PERCENTAGE
    sim = csr_matrix(sim)
    sim.setdiag(0)
    sim = 1*(sim>=thr)
    sim = 1*((sim+sim.T)>=1)
    clique_size = sim.sum(axis=1).A.flatten()

    if size:
        return np.mean(clique_size+1)
    else:
        return sum(clique_size>0)