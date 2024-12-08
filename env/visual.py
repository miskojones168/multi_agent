# Visualization of a MOS instance using pygame
#
# Note to run this file, you need to run the following
# in the parent directory of multi_object_search:
#
#   python -m multi_object_search.env.visual
#

import pygame
import cv2
import math
import numpy as np
import random
import pomdp_py.utils as util

import env.env as e
import domain.observation as o
import domain.action as a
import domain.state as s
import example_worlds as exw
import time
import datetime

# Deterministic way to get object color
def object_color(objid, count):
    color = [107, 107, 107]
    if count % 3 == 0:
        color[0] += 100 + (3 * (objid * 5 % 11))
        color[0] = max(12, min(222, color[0]))
    elif count % 3 == 1:
        color[1] += 100 + (3 * (objid * 5 % 11))
        color[1] = max(12, min(222, color[1]))
    else:
        color[2] += 100 + (3 * (objid * 5 % 11))
        color[2] = max(12, min(222, color[2]))
    return tuple(color)

#### Visualization through pygame ####
class MosViz:
    def __init__(self, env, res=30, fps=30, controllable=False):
        self._env = env

        self._res = res
        self._img = self._make_gridworld_image(res)
        self._last_observation = {}  # map from robot id to MosOOObservation
        self._last_viz_observation = {}  # map from robot id to MosOOObservation
        self._last_action = {}  # map from robot id to Action
        self._last_belief = {}  # map from robot id to OOBelief

        self._controllable = controllable
        self._running = False
        self._fps = fps
        self._playtime = 0.0

        # Generate some colors, one per target object
        colors = {}
        for i, objid in enumerate(env.target_objects):
            colors[objid] = object_color(objid, i)
        self._target_colors = colors

    def _make_gridworld_image(self, r=30):
        # Preparing 2d array
        w, l = self._env.width, self._env.length
        arr2d = np.full((self._env.width, self._env.length), 0)  # free grids
        state = self._env.state
        dropped_boxes = 0
        in_station = []
        boxes = state.box_states()
        stations = state.station_states()
        for bid, b in boxes:
            for sid, st in stations:
                if b.pose == st.pose:
                    dropped_boxes += 1
                    in_station.append(bid)

        for objid in state.object_states:
            pose = state.object_states[objid]["pose"]
            if state.object_states[objid].objclass == "obstacle":
                arr2d[pose[0], pose[1]] = 1  # obstacle
            elif state.object_states[objid].objclass == "station":
                arr2d[pose[0], pose[1]] = 3  # station
            elif state.object_states[objid].objclass == "box" and type(state.state(objid).carrier_id) == type(None) and objid not in in_station:
                arr2d[pose[0], pose[1]] = 2  # target
            

        # Creating image
        img = np.full((w * r, l * r, 3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                if arr2d[x, y] == 0:  # free
                    cv2.rectangle(
                        img, (y * r, x * r), (y * r + r, x * r + r), (255, 255, 255), -1
                    )
                elif arr2d[x, y] == 1:  # obstacle
                    cv2.rectangle(
                        img, (y * r, x * r), (y * r + r, x * r + r), (40, 31, 3), -1
                    )
                elif arr2d[x, y] == 3:  # station
                    top_left, bottom_right = (y * r, x * r), (y * r + r, x * r + r)
                    cv2.rectangle(
                        img, (y * r, x * r), (y * r + r, x * r + r), (165, 255, 165), -1
                    )
                    # Calculate center of the rectangle
                    rect_center_x = (top_left[0] + bottom_right[0]) // 2
                    rect_center_y = (top_left[1] + bottom_right[1]) // 2
                    # Prepare text for the number
                    text = str(dropped_boxes)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    text_thickness = 2
                    # Calculate text size
                    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
                    text_x = rect_center_x - text_size[0] // 2  # Center the text horizontally
                    text_y = rect_center_y + text_size[1] // 2  # Center the text vertically
                    # Draw the text
                    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

                elif arr2d[x, y] == 2:  # box
                    cv2.rectangle(
                        img, (y * r, x * r), (y * r + r, x * r + r), (255, 165, 0), -1
                    )

                cv2.rectangle(
                    img, (y * r, x * r), (y * r + r, x * r + r), (0, 0, 0), 1, 8
                )
        return img

    @property
    def img_width(self):
        return self._img.shape[0]

    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def last_observation(self):
        return self._last_observation

    def update(self, robot_id, action, observation, viz_observation, belief):
        """
        Update the visualization after there is new real action and observation
        and updated belief.

        Args:
            observation (MosOOObservation): Real observation
            viz_observation (MosOOObservation): An observation used to visualize
                                                the sensing region.
        """
        self._last_action[robot_id] = action
        self._last_observation[robot_id] = observation
        self._last_viz_observation[robot_id] = viz_observation
        self._last_belief[robot_id] = belief

    @staticmethod
    def draw_robot(img, x, y, th, size, id, color=(255, 12, 12)):
        radius = int(round(size / 2))
        cv2.circle(img, (y + radius, x + radius), radius, color, thickness=2)

        endpoint = (
            y + radius + int(round(radius * math.sin(th))),
            x + radius + int(round(radius * math.cos(th))),
        )
        cv2.line(img, (y + radius, x + radius), endpoint, color, 2)

    @staticmethod
    def draw_belief(img, belief, r, size, target_colors):
        """belief (OOBelief)"""
        radius = int(round(r / 2))

        circle_drawn = {}  # map from pose to number of times drawn

        for objid in belief.object_beliefs:
            if isinstance(belief.object_belief(objid).random(), s.RobotState):
                continue
            hist = belief.object_belief(objid).get_histogram()
            color = target_colors[objid]

            last_val = -1
            count = 0
            for state in reversed(sorted(hist, key=hist.get)):
                if state.objclass == "target":
                    if last_val != -1:
                        color = util.lighter(color, 1 - hist[state] / last_val)
                    if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.99:
                        tx, ty = state["pose"]
                        if (tx, ty) not in circle_drawn:
                            circle_drawn[(tx, ty)] = 0
                        circle_drawn[(tx, ty)] += 1

                        cv2.circle(
                            img,
                            (ty * r + radius, tx * r + radius),
                            size // circle_drawn[(tx, ty)],
                            color,
                            thickness=-1,
                        )
                        last_val = hist[state]

                        count += 1
                        if last_val <= 0:
                            break

    # PyGame interface functions
    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode(
            (self.img_width, self.img_height), pygame.HWSURFACE
        )
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont("Comic Sans MS", 30)
        self._running = True

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0

    def on_render(self, iii=0):
        self.render_env(self._display_surf, iii=iii)
        for rid in list(self._env.robot_ids):
            rx, ry, rth = self._env.state.pose(rid)
            fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
            last_action = self._last_action.get(rid, None)
            last_action_str = "no_action" if last_action is None else str(last_action)
            pygame.display.set_caption(
                "%s | Robot%d(%.2f,%.2f,%.2f) | %s"
                % (
                    last_action_str,
                    rid,
                    rx,
                    ry,
                    rth * 180 / math.pi,
                    fps_text,
                )
            )
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def render_env(self, display_surf, iii=0):
        # draw robot, a circle and a vector
        img = np.copy(self._img)
        for i, robot_id in enumerate(self._env.robot_ids):
            rx, ry, rth = self._env.state.pose(robot_id)
            is_loaded = True if type(self._env.state.state(robot_id).load) != type(None) else False
            r = self._res  # Not radius!
            last_observation = self._last_observation.get(robot_id, None)
            last_viz_observation = self._last_viz_observation.get(robot_id, None)
            last_belief = self._last_belief.get(robot_id, None)
            if last_belief is not None:
                MosViz.draw_belief(img, last_belief, r, r // 3, self._target_colors)

            color = (12, 255 * (0.8 * (i + 1)), 12) if not is_loaded else (200, 12, 12)
            MosViz.draw_robot(
                img, rx * r, ry * r, rth, r, color=color, id=str(i)
            )
        pygame.surfarray.blit_array(display_surf, img)


if __name__ == "__main__":
    pass
