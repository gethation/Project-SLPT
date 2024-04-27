condaimport os
from utils import compose
import json

if __name__ == "__main__":
    while True:
        with open('schedule.json', 'r') as f:
            schedule = json.load(f)

        index = schedule['progressive']
        paths = schedule['list']
        if index <= len(paths)-1:
            schedule['progressive'] += 1

            with open('schedule.json', 'w') as f:
                json.dump(schedule, f)

            compose(paths[index])
        else:
            print('done')
            break