import pandas as pd
import os

from data_process.arch.trip_stop_schedule_check import *

MAIN_DATA_DIR = "transsee_spider/data/ttc/"
WEB_LINK_TMPL = 'https://www.transsee.ca/triplist?a=ttc&t=route&route={route}&date={date}&starttime=04%3A00&endtime=04%3A00&nextday=on&ok=OK'
ROUTE = "29"
DATE = "2023-01-01"

def return_agent_schedule_for_route_at_date(route, date):
    root = os.path.join(MAIN_DATA_DIR, route)
    root = os.path.join(root, date)
    file_path = os.path.join(root, f"{route}_{date}_agent_schedule.csv")
    web_ref_link = WEB_LINK_TMPL.format(route=route, date=date)
    return file_path, web_ref_link

def read_agent_schedule():
    agent_schedule = pd.read_csv(os.path.join(MAIN_DATA_DIR, f"{ROUTE}_agent_schedule.csv"))
    return agent_schedule

def main():
    file_path, web_ref_link = return_agent_schedule_for_route_at_date(ROUTE, DATE)
    print("Web reference link:", web_ref_link, end="\n\n")
    agent_schedule = pd.read_csv(file_path)

    print("Num of Rows:", agent_schedule.shape[0])
    print("Num of Columns without Destination:", agent_schedule[agent_schedule["destination"].isnull()].shape[1])

    print("Rows with missing information:", agent_schedule[agent_schedule.isnull().any(axis=1)].shape[0])

    # Iterate over all rows with missing information, print the columns with missing information, trip_id, and trip_sched_id
    missing_info = agent_schedule[agent_schedule.isnull().any(axis=1)]
    count = 0
    for idx, row in missing_info.iterrows():
        count += 1
        print("Row:", count)
        print("Trip ID:", row["trip_id"])
        print("Trip Schedule ID:", row["trip_sched_id"])
        print("Columns with missing information:", row[row.isnull()].index.tolist())
        print()

if __name__ == "__main__":
    main()
