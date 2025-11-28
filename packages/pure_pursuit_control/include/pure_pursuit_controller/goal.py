import numpy as np

def pt_to_pt_distance (pt1,pt2):
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance

def sgn (num):
    return 1 if num >= 0 else -1

def find_goal_point(path, current_pos, lookahead_distance, last_found_index):
    """
    Finding goal point from path using the line-circle intersection formula.
    Docs:
    - https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit

    4 Cases:
    1) no intersection => discriminant is negative and we need to fallback 
      to default goal point. TBD
    2) intersection found, but not within (x1, y1) and (x2, y2) => consider 
      this case as no intersection
    3) one intersection within (x1, y1) and (x2, y2) => select that point
    4) two intersections within (x1, y1) and (x2, y2) => we select the best of the two points

    Parameters
    ----------

    Returns
    -------
    goal_point: Tuple[float, float]
    last_found_index: int

    Notes
    -----
    Handling the case when no intersection is found: TODO
    - Extend lookahead_distance with max
    - Straight Line to closest point along path
    """
    intersection_found = False
    starting_index = last_found_index

    # Note: default goal point is last point in path - FIXME: can we trust this?
    goal_point = [path[-1][0], path[-1][1]] 

    if len(path) == 1:
        goal_point = [path[0]]
        last_found_index = 0
        return goal_point, last_found_index

    for i in range(starting_index, len(path) -1):
        #  if intersection_found:
        #      return goal_point, last_found_index

        x1, x2 = path[i][0] - current_pos[0], path[i+1][0] - current_pos[0]
        y1, y2 = path[i][1] - current_pos[1], path[i+1][1] - current_pos[1]
        dx, dy = x2 - x1, y2 - y1
        dr = np.sqrt(dx**2 + dy**2)
        D = x1*y2 + x2*y1
        discriminant = (lookahead_distance**2) * (dr**2) - D**2

        if discriminant >= 0:
            sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
            sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
            sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

            sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]

            minX = min(path[i][0], path[i+1][0])
            minY = min(path[i][1], path[i+1][1])
            maxX = max(path[i][0], path[i+1][0])
            maxY = max(path[i][1], path[i+1][1])

            is_sol1_in_range = ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY))
            is_sol2_in_range = ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY))

            if not is_sol1_in_range and not is_sol2_in_range:
                intersection_found = False
                goal_point = [path[last_found_index][0], path[last_found_index][1]]
            if is_sol1_in_range and not is_sol2_in_range:
                intersection_found = True
                goal_point = sol_pt1
            elif not is_sol1_in_range and is_sol2_in_range:
                intersection_found = True
                goal_point = sol_pt2
            # case 4: both points in range => select point closest to last path point
            elif is_sol1_in_range and is_sol2_in_range:
                intersection_found = True
                goal_point = sol_pt1 if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1])

            if intersection_found and pt_to_pt_distance(goal_point, path[i+1]) < pt_to_pt_distance(current_pos, path[i+1]):
                last_found_index = i+1
                break


        # TODO: what to do if no intersection found
        if not intersection_found:
            pass

        return goal_point, last_found_index


