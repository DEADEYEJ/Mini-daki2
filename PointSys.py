import Tilegridder as TG

image_path = r"king Domino dataset\1.jpg"

# Temp CrownGrid
def get_crown_grid():
    return [[0,0,0,0,0],
            [0,0,0,1,0],
            [0,1,0,0,0],
            [0,2,0,2,1],
            [0,1,0,1,0]]

# DFS to find connected regions of the same terrain type
def find_regions(terrain_grid):
    visited = [[False]*5 for _ in range(5)]
    regions = []

    def dfs(y, x, terrain):
        stack = [(y, x)]
        tiles = []

        while stack:
            cy, cx = stack.pop()
            if visited[cy][cx]:
                continue
            if terrain_grid[cy][cx] != terrain:
                continue

            visited[cy][cx] = True
            tiles.append((cy, cx))

            # 4-directional neighbors
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < 5 and 0 <= nx < 5:
                    stack.append((ny, nx))
        return tiles

    for y in range(5):
        for x in range(5):
            if not visited[y][x]:
                terrain = terrain_grid[y][x]
                region_tiles = dfs(y, x, terrain)
                regions.append((terrain, region_tiles))
    return regions


def score_regions(regions, crown_grid):
    total_score = 0

    for terrain, tiles in regions:
        size = len(tiles)
        crowns = sum(crown_grid[y][x] for y, x in tiles)
        score = size * crowns

        #print(f"{terrain}: size={size}, crowns={crowns}, score={score}")

        total_score += score
    return total_score

    
def main():
    terrain = TG.terrain_grid(image_path)
    crowns = get_crown_grid()

    regions = find_regions(terrain)
    total_score = score_regions(regions, crowns)

    print("Total score:", total_score)



if __name__ == "__main__":
    main()