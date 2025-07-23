import os
import rl_games
print(f"rl_games 路徑: {os.path.dirname(rl_games.__file__)}")
print(f"查找 players.py 文件:")
for root, dirs, files in os.walk(os.path.dirname(rl_games.__file__)):
    for file in files:
        if file == "players.py":
            print(f"找到: {os.path.join(root, file)}")