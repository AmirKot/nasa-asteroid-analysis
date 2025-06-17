"""
@Project: Maman15

@Description : Preparing, Analysis,show a data from file about Asteroids.

@Author: Amir Kot
@semester : 2024b
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data(file_name):
    '''
    Load the data of the file  and return it in  np.ndarray form
    @param:
          file_name (str): the name of the file, the file must be csv
    @Returns:
           The file with as np.ndarray
    '''
    try:
        data = np.genfromtxt(file_name, delimiter=',', encoding='utf-8', dtype=None)
        data = np.array(data)
        return data
    except OSError:
        raise  OSError(f"Unable to open file {file_name}.")  # maybe use raise


def scoping_data(data, names):
    '''
     It removes the columns in data that is header(name) is in names
     and retuen the new data
      @param:
            data (np.ndarray): the data in its first row is the headers (colmn name)
            names (list of str): the list of the colmn names to be remobed from the data
      @return:
               data without th columns that its name in the list names
    '''
    if type(names) != list or type(data) != np.ndarray:
        raise TypeError("The arguments (data,names) should be type (ndarray,list)")
    elif data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    # make list in each index represnt if the colmn stay in the retuen data(yes=True,no= False
    list_good_inex =[True for i in range(data.shape[1])]

    for i in range(data.shape[1]):
        if data[0,i] in names: # the colmn name is in the fisrt row
            list_good_inex[i] = False
    return  data[:,list_good_inex]

def mask_data(data):
    '''
    It updates and return the data to have only the rows which it's the (Close Approach Date) column is from 2000 and later
    @param:
          data (np.ndarray): The data to be updated
    @return:
           return the data only with the rows that the value in colmn[Close Approach Date] if from 2000 and later
    '''
    if type(data) != np.ndarray or data.ndim!=2 :
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")

    colmn_name = 'Close Approach Date' # the column in data
    year = '2000' # the year to check as str

    # check if the colmn exist
    if not(colmn_name in data[0,:]):
        raise  ValueError(f"The colmn {colmn_name} is not in the data")

    index_colmn =np.where(data[0]==colmn_name)[0][0]  # get the index of the column
    #make mask (list) to represent each row, the value in boolean (True=The year in column after the(year),False=otherwise)
    mask = [if_after_year(year,data[i,index_colmn]) for  i in range(1,data.shape[0])]
    #add True to include the headers  (header in row 0)
    mask.append(True)
    mask.reverse()
    new_data = data[mask,:]
    return new_data
def if_after_year(year1,date):
    '''
    check if the date is in that year or later and return boolean value
    @param:
        year1 (str): The year to check
        date (str): The date in form yyyy-mm-dd or yyyy-dd-mm
    @return:
        return True if the date in The year(year) or later ,False other wisex
    '''
    return int(year1) <= int((str(date).split('-'))[0])


def data_details(data):
    '''
    Removes the column named:'Neo Reference ID', 'Orbiting Body', 'Equinox' in data and return it
    and print the number of rows,column and headers in data
    @param:
        data (ndarray): The data to remove the column , it should be a matrix(2D)
    @return:
        return the data without the columns that it name: 'Neo Reference ID', 'Orbiting Body', 'Equinox'
    '''
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    list_of_column_delete = ['Neo Reference ID', 'Orbiting Body', 'Equinox']
    new_data = scoping_data(data,list_of_column_delete)  # Update the data removing the columns

    print(f"The number of rows in data:{new_data.shape[0]} \nThe number of columns in data:{new_data.shape[1]}")
    print(f"The current headers in data:\n{new_data[0,:]}")
    return new_data

def max_absolute_magnitude(data):
    '''
      Return the name and value of asroude that have the max Absolute Magnitude in relation to Earth
    @param:
        data (ndarray): 2 dimension data , that have column 'Absolute Magnitude' and 'Name'
    @return:
       tuple of the withe first val is the name of asteroid that have max 'Absolute Magnitude'
        ,the second val is the value of the 'Absolute Magnitude'
    '''
    # Check the data if Have columns named : 'Absolute Magnitude' , 'Name'
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not ( ('Absolute Magnitude' in data[0, :] )and ('Name' in data[0, :])):
        raise ValueError("The argument data should have columns with names is (Absolute Magnitude) and(Name) ")

    #The columns name
    col_absulute ='Absolute Magnitude'
    col_name = 'Name'

    #The inex of the columns
    index_column_absu = np.where(data[0,:]== col_absulute)[0][0]
    index_column_name = np.where(data[0,:]== col_name)[0][0]

    # Get the max of the in column col_absulute as type float
    max_val_index_row =1 # in data[0:] is the headers
    for i in range(1,index_column_absu):  # in data[0:] is the headers
        try:
            curr_vall = float(data[i, index_column_absu])
            if curr_vall > float(data[max_val_index_row, index_column_absu]):
                max_val_index_row = i
        except ValueError:
            raise ValueError(f"can't convert the value{curr_vall}or {data[max_val_index_row,index_column_absu]} to float in column {index_column_absu}")

    max_val = float(data[max_val_index_row, index_column_absu])
    return ( data[max_val_index_row, index_column_name], max_val )


def closest_to_earth(data):
    '''
    Return the name of the closest asteroid to Earth according to column  'Miss Dist.(kilometers)'
    @param:
        data (ndarray): 2 dimension data , that have column 'Miss Dist.(kilometers)' and 'Name'
    @return:
          Return the name of the closest asteroid to Earth according to column 'Miss Dist.(kilometers)'
    '''
    # Check the data if Have columns named : 'Miss Dist.(kilometers)' , 'Name'
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not ( ('Absolute Magnitude' in data[0,:] )and ('Name' in data[0,:]) ) :
        raise ValueError("The argument data should have columns with names is (Miss Dist.(kilometers)) and(Name) ")

    #colmn names
    col_Dis_name ='Miss Dist.(kilometers)'
    col_name = 'Name'

    #column inex
    col_Dis_index = np.where(data[0,:]==col_Dis_name)[0][0]
    col_name_index =np.where(data[0,:] == col_name)[0][0]

    #Get the min of the in column col_Dis_index as type float
    min_val_index_row =1 # in data[0:] is the headers
    for i in range(1,data.shape[0]):
        try:
            curr_vall =float(data[i,col_Dis_index])
            if curr_vall < float( data[min_val_index_row,col_Dis_index]) :
                min_val_inex_row =i
        except ValueError:
            raise ValueError(f"can't convert the value{curr_vall}or {data[min_val_index_row,col_Dis_index]} to float in column {col_Dis_index}")

    return data[min_val_inex_row, col_name_index]

def common_orbit(data):
    '''
    Make dict with key as ('Orbit ID') and value the number of asteroid in that orbit
    @param:
        data (ndarray): 2 dimension data , that have column 'Orbit ID'
    @return:
       dict with key as Orbit ID and the value how many asteroid in that orbit according to data
    '''
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not ('Orbit ID' in data[0,:]):
        raise ValueError("The argument data should have columns with name ('Orbit ID')")

    #get the column name
    col_name = 'Orbit ID'
    col_index = np.where(data[0,:]==col_name)[0][0]

    #All the Orbit ID as array
    List=[]
    for i in range(1,data.shape[0]):
        try:
            List.append(int(data[i,col_index]))
        except:
            raise ValueError(f"Can't convert the value {data[i,col_index]} to int")
    arr_ID = np.array(List)
    arr_ID = np.unique(arr_ID) #migh be some Id repeted more then one

    dic = dict.fromkeys(arr_ID,0)

    for key in dic.keys():
        condition = data[:,col_index] == str(key)
        dic[key] = len( data[condition, col_index]) # count how many asteroid have the same key(Orbit Id)
    return dic

def min_max_diameter(data):
    '''
    Calculate the average of the columns(min\max diameter of asteroid) :(Est Dia in KM(min)) and (Est Dia in KM(max)) in data)
    @param:
        data (ndarray): 2 dimension data , that have column 'Est Dia in KM(min)' and 'Est Dia in KM(max))' in data
    @return:
         Tuple with first value the average of min diameters , the second value the average of the max diameters
    '''
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not (('Est Dia in KM(min)' in data[0,:]) and ('Est Dia in KM(max)' in data[0,:])):
        raise ValueError("The argument data should have columns with name (Est Dia in KM(min)) and (Est Dia in KM(max))")
    #columns name
    min_col_name = 'Est Dia in KM(min)'
    max_com_name = 'Est Dia in KM(max)'

    # Get colmun index
    min_col_index = np.where(data[0,:] ==min_col_name)[0][0]
    max_col_index = np.where(data[0,:]==max_com_name)[0][0]

    #make list of the min ,max dim
    min_list, max_list = [], []
    for i in range(1,data.shape[0]):
        try:
            min_list.append(float(data[i, min_col_index]))
            max_list.append(float(data[i, max_col_index]))
        except ValueError:
            raise ValueError(f"Can't convert the value{data[i, min_col_index]} or{data[i, min_col_index]} to float")
    #calculate the min, max average
    min_avg = sum(min_list)/(data.shape[0]-1) # -1 because the first row is should
    max_avg = sum(max_list)/(data.shape[0]-1) # -1 because the first row is should
    return (min_avg,max_avg)

def plt_hist_diameter(data):
    '''
    Make and show/print a Histogram of Average(Est Dia in KM(min),Est Dia in KM(max)) Diameter of Asteroids in data
    @param
        data (ndarray): 2 dimension data , that have column 'Est Dia in KM(min)' and 'Est Dia in KM(max))' in data
    @return:
          None -doesn't return but show/print diagram
    '''
    # check the same error for data as (min_max_diameter())
    try:
        # get the range
        diameter_range = min_max_diameter(data)
    except (TypeError,ValueError) as err:
        raise err
    min_val, max_val = diameter_range[0] ,diameter_range[1]

    # columns name
    min_col_name = 'Est Dia in KM(min)'
    max_com_name = 'Est Dia in KM(max)'

    # Get colmun index
    min_col_index = np.where(data[0, :] == min_col_name)[0][0]
    max_col_index = np.where(data[0, :] == max_com_name)[0][0]

    #make list of all the avrage diamters in range
    List= []
    c=0
    for i in range(1, data.shape[0]):
        avrage =(float(data[i,min_col_index])+float(data[i,max_col_index]))/2 # if the value can't convert to float  an Error will rais in min_max_diameter(data)
        if avrage <= max_val and min_val <= avrage:
            List.append(avrage)


    arr = np.array(List)

    #make the histogram
    plt.hist(arr,bins=10,color='steelblue',edgecolor='black')
    plt.title("Histogram of Average Diameter of Asteroids")
    plt.xlabel("Average Diameter (Km)")
    plt.ylabel("Number of Asteroids")
    plt.grid(True)
    plt.show()

def plt_hist_common_orbit(data):
    '''
     Make and show/print a Histogram of Asteroids by Minimum Orbit Intersection
    @param
        data (ndarray): 2 dimintion data , that have column 'Est Dia in KM(min)' and 'Est Dia in KM(max))' in data
    @return:
        None -dosen't return but show/print diagram
    '''
    # check the same error for data as (common_orbit())
    try:
        #get the orbit Id and number of Asteroids in it
        dic =common_orbit(data)
    except (TypeError, ValueError) as err:
        raise err

    #List the have for each Asteroids its Orbit ID as value
    List = []
    for key in dic.keys():
        List += [key for i in range(dic[key])]

    # make the histogram
    plt.hist(List,bins=6,color='steelblue',edgecolor='black')
    plt.title('Histogram of Asteroids by Minimum Orbit Intersection')
    plt.xlabel('Minimum Orbit Intersection')
    plt.ylabel('Number of Asteroids')
    plt.grid(True)
    plt.show()

def plt_pie_hazard(data):
    '''
    print/show pie diagram of the percentage Hazardous and the None-Hazardous  Asteroids
    @param
        data (ndarray): 2 dimension data , that have column 'Hazardous' in data
    @return:
        None -doesn't return but show/print diagram
    '''
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not ('Hazardous' in data[0,:]):
        raise ValueError("The argument data should have columns with name (Hazardous)")

    #get the column name and index
    column_name = 'Hazardous'
    column_index = np.where(data[0,:]==column_name)[0][0]

    #Count how many astoueds are Hazardous in data
    None_hazard_num = len(np.where(data[:, column_index] == 'False')[0])
    hazard_num = len(np.where(data[:, column_index] == 'True')[0])

    #make Diagram
    plt.pie([hazard_num,None_hazard_num],colors=['red','green'],labels=['Hazardous','Non-Hazardous'],autopct='%1.1f%%')
    plt.title('Percentage of Hazardous and None-Hazardous Asteroids')
    plt.show()

def plt_liner_motion_magnitude(data):
    '''
    print/show grid of diagram Linear Relationship between (Absolute Magnitude) and (Miles per hour) if there is one
    @param
        data (ndarray): 2 dimension data , that have columns 'Absolute Magnitude' and 'Miles per hour'  in data
    @return:
        None -doesn't return but might show/print grid
    '''
    if type(data) != np.ndarray or data.ndim != 2:
        raise TypeError("The arguments (data) should be type (ndarray) and 2-dimension")
    if not (('Absolute Magnitude' in data[0,:]) and ('Miles per hour' in data[0,:])):
        raise ValueError("The argument data should have columns with name (Absolute Magnitude) and (Miles per hour)")

    # get the column name
    magnitude_col_name = 'Absolute Magnitude'
    motion_col_name = 'Miles per hour'

    # get the column index
    magnitude_col_index = np.where(data[0, :] == magnitude_col_name)[0][0]
    motion_col_index = np.where(data[0, :] == motion_col_name)[0][0]

    X = np.array([float(da)for da in data[1:, magnitude_col_index]]) # x values are the Absolute Magnitude
    Y = np.array([float(da)for da in data[1:, motion_col_index]]) # y values are the Miles per hour

    a,b,r_val,p_val,std_err = stats.linregress(X, Y)
    if p_val < 0.05:
        plt.scatter(X, Y)
        plt.plot(X, a*X +b,color='red')
        plt.title('Linear Relationship between (Absolute Magnitude) and (Miles per hour)')
        plt.xlabel('Absolute Magnitude')
        plt.ylabel('Miles per hour')
        plt.legend(['Data points',f'Fitted line (r={r_val:.2f})'],loc= 'upper right')
        plt.grid(True)
        plt.show()

# below is the  main & it's help functios

def main():
    '''
        Create a main for The program of Analysing Asteroids
        '''
    # Get  file name
    file_name = 'nasa.csv'

    try:
        data = load_data(file_name)
        # Graphs of orignal data
        print("This are the graph without changing data:")
        # plt_hist_diameter
        plt_hist_diameter(data)
        # plt_hist_common_orbit
        plt_hist_common_orbit(data)
        # plt_pie_hazard
        plt_pie_hazard(data)
        # plt_liner_motion_magnitudeplt_liner_motion_magnitude(data)
        plt_liner_motion_magnitude(data)
        print("If there is linear Relationship => graph will show")


        # Graphs of updated data
        data = mask_data(data)
        data = data_details(data)
        tup = max_absolute_magnitude(data)
        print('(name,valus)  of Asteroids the  have Maximum absolute magnitude: ',tup)
        print('the name of closest Asteroids to earth by (in Km): ',closest_to_earth(data))
        tup = min_max_diameter(data)
        print('(avg_min,avg_max) => minimun, maximum average radius for Asteroids: ',tup)

        print("This are the graph with updated data:")
        # plt_hist_diameter
        plt_hist_diameter(data)
        # plt_hist_common_orbit
        plt_hist_common_orbit(data)
        # plt_pie_hazard
        plt_pie_hazard(data)
        # plt_liner_motion_magnitudeplt_liner_motion_magnitude(data)
        plt_liner_motion_magnitude(data)
        print("If there is linear Relationship => graph will show")
    except (OSError,TypeError,ValueError) as err:
        print('Error : ', err)



main()


#below there is interactive main if ot was needed  ...

'''

def print_menu():
    print("### Menu ###")
    print("Enter 1: to change file/data") # load_data
    print("Enter 2: to delete columns") # scoping_data
    print("Enter 3: Update to Only have Asteroids approach from 2000 and after") # mask_data
    print("Enter 4: Delete The colmuns named ('Neo Reference ID', 'Orbiting Body', 'Equinox') and show the headers and data dimantions")# data_details
    print("Enter 5: Get tuple get (name,valus)  of Asteroids the  have Maximum absolute magnitude ")# max_absolute_magnitude
    print("Enter 6: Get the name of closest Asteroids to earth by (in Km)")# closest_to_earth
    print("Enter 7: Get dictinoary each vale is {Orbit ID: number of Asteroids in orbit}")# common_orbit
    print("Enter 8: Get tuple (avg_min,avg_max) => minimun, maximum average radius for Asteroids ")# min_max_diameter
    print("Enter 9: Histogram number of Asteroids radius average (10 bins)")# plt_hist_diameter
    print("Enter 10: Histogram of Asteroids by Minimum Orbit Intersection{continuous} (6 bins)")# plt_hist_common_orbit
    print("Enter 11: Pie graph of Percentage  Hazardous/None-Hazardous Asteroids ")# plt_pie_hazard
    print("Enter 12: Make diagram if there is linear Relationship between (Absolute Magnitude) and (Miles per hour)")# plt_liner_motion_magnitude
    print("Enter Quit: to exit")


def get_func_val():
    ''' """
    Get value in the Menu from that shown in function 'print_menu()'
    @return:
     The value from the Menu
    '''
    while(True):
        val = input("Enter value in Menu ")
        if val.isnumeric() and int(val)<=12 and 1<=int(val):
            return val
        elif val == 'Quit':
            return val
        else:
            print("!!!!!!! This input is not correct ")


def main():
    
    #Get first file name
    while True:
        try:
            file_name = input("Enter a file name to start :")
            data = load_data(file_name)
            break
        except OSError as err:
            print('Error : ', err)

    # Print menu
    print_menu()

    continue_Main =True
    while continue_Main:
        func = get_func_val()
        # Preparing data
        try:
            # load_data
            if func == '1':
                file_name = input("Enter a file name:")
                data = load_data(file_name)
                continue
            # scoping_data
            elif func == '2':
                List = input("Enter list of column name divided by ',' :").split(',')
                List = [name.strip() for name in List]
                data = scoping_data(data, List)
                continue
            # mask_data
            elif func == '3':
                data = mask_data(data)
                continue
            # data_details
            elif func == '4':
                data = data_details(data)
                continue
        except (OSError ,TypeError , ValueError) as err:
            print("Error : ",err)

        # Analysis data
        try:
            # max_absolute_magnitude
            if func == '5':
                tup = max_absolute_magnitude(data)
                print("The tuple : ", tup)
                continue
            # closest_to_earth
            elif func == '6':
                name = closest_to_earth(data)
                print("The name of the closest Asteroids: ", name)
                continue
            # common_orbit
            elif func == '7':
                dic = common_orbit(data)
                print("The dictonary :\n", dic)
                continue
            # min_max_diameter
            elif func == '8':
                tup = min_max_diameter(data)
                print("The tuple : ", tup)
        except (TypeError,ValueError) as err:
            print("Error : ", err)

        # show graph
        try:
            # plt_hist_diameter
            if func == '9':
                plt_hist_diameter(data)
            # plt_hist_common_orbit
            elif func == '10':
                plt_hist_common_orbit(data)
            # plt_pie_hazard
            elif func == '11':
                plt_pie_hazard(data)
            # plt_liner_motion_magnitude
            elif func == '12':
                plt_liner_motion_magnitude(data)
                print("If there is linear Relationship => graph will show")
            elif func == 'Quit':
                continue_Main = False
        except (TypeError,ValueError) as err:
            print("Error : ", err)

''' """
