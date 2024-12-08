"""This file has some examples of world string."""

import random

############# Example Worlds ###########
# See env.py:interpret for definition of
# the format

world0 = (
    """
rx..o
.x.xT
.....
""",
    "r",
)

world1 = (
    """
rx.T..o
.x.....
...xx..
.......
.xxx.T.
.xxx...
.......
""",
    "r",
)

# Used to test the shape of the sensor
world2 = (
    """
.................
.................
..xxxxxxxxxxxxxo.
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxTxxxx..
..xxxxxxrxTxxxx..
..xxxxxxxxTxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""",
    "r",
)

# Used to test sensor occlusion
world3 = (
    """
.................
.................
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxTxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxx..oxxxxxx..
..xxxx..xx.xxxx..
..xxxx..r.Txxxx..
..xxxx..xx.xxxx..
..xxxxxx..xxxxx..
..xxxxTx..xxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxxxxxxxxxxx..
.................
.................
""",
    "r",
)

world4 = (
    """
.................
..xxxTxxxxxxxxx..
..xxxxxxxxxxxxx..
..xxxx..oxxxxxx..
..xxxx..xx.xxxx..
..xxxxxxxrTxxxx..
..xxxxTxxxxxxxx..
.................
""",
    "r",
)

world5 = (
    """
.xxxx..o.xxxxx.
.xxxx..x....xx.
.xx....r.Tx.xx.
.xx.x.....x.xx.
.xx.x......Txx.
.xx.xTx..xxxxx.
.xx.........T..
.x..x.xax.xxxx.
.xxxxox...xxxx.
""",
    "r", 'a',
)

def random_world(width, length, num_obj, num_obstacles, num_stations = 1, robot_char="r"):
    worldstr = [["." for i in range(width)] for j in range(length)]
    # First place obstacles
    num_obstacles_placed = 0
    while num_obstacles_placed < num_obstacles:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "x"
            num_obstacles_placed += 1

    num_obj_placed = 0
    while num_obj_placed < num_obj:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "T"
            num_obj_placed += 1
    
    num_stations_placed = 0
    while num_stations_placed < num_stations:
        x = random.randrange(0, width)
        y = random.randrange(0, length)
        if worldstr[y][x] == ".":
            worldstr[y][x] = "o"
            num_stations_placed += 1

    # Finally place the robot
    # for cha in ['r', 'a']:
    #     while True:
    #         x = random.randrange(0, width)
    #         y = random.randrange(0, length)
    #         if worldstr[y][x] == ".":
    #             worldstr[y][x] = cha
    #             break
    
    while True:
            x = random.randrange(0, width)
            y = random.randrange(0, length)
            if worldstr[y][x] == ".":
                worldstr[y][x] = robot_char
                break

    # Create the string.
    finalstr = []
    for row_chars in worldstr:
        finalstr.append("".join(row_chars))
    finalstr = "\n".join(finalstr)
    # return finalstr, 'r', 'a'
    return finalstr, robot_char
