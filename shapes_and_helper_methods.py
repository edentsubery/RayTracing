import math
import numpy as np

Epsilon = 0.0000001


def vector_normal(vector):
    return vector / np.linalg.norm(vector)


class Ray:
    def __init__(self, source, direction):
        self.source = source
        self.direction = direction

    def nearest_intersection(self, objects):
        nearest_obj = None
        min_dist = np.inf
        for obj in objects:
            curr_obj, curr_dist = obj.intersect(self)
            if curr_dist is not None:
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    nearest_obj = curr_obj

        return nearest_obj, min_dist


class Light:

    def __init__(self, position, light_color, specular_intensity, shadow_intesity, radius):
        self.position = np.array(position)
        self.color = np.array(light_color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intesity
        self.radius = radius


class Shape:

    def __init__(self):
        self.transparency = None
        self.phong = None
        self.specular = None
        self.diffuse = None
        self.reflection = None

    def material(self, diffuse, specular, reflection, phong, transparency):
        self.diffuse = diffuse
        self.specular = specular
        self.reflection = reflection
        self.phong = phong
        self.transparency = transparency


class Plane(Shape):
    def __init__(self, normal, offset):
        super().__init__()
        self.normal = np.array(normal)
        self.offset = offset

    def intersect(self, ray: Ray):
        N = self.normal
        denominator = np.dot(ray.direction, N)
        if denominator == 0:
            return None, math.inf
        nominator = -1 * np.dot(ray.source, N) + self.offset
        if nominator < Epsilon and nominator > -Epsilon:
            nominator = 0
        t = nominator / denominator
        if t > 0:
            return self, t
        return None, math.inf

    def normalized(self, intersection_point):
        return np.linalg.norm(self.normal)

    def get_normal(self, intersection_point):
        return self.normal


class Sphere(Shape):
    def __init__(self, center, radius: float):
        super().__init__()
        self.center = np.array(center)
        self.radius = radius

    def get_normal(self, intersection_point):
        return vector_normal(intersection_point - self.center)

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)

    def intersect(self, ray: Ray):
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, (ray.source - self.center))
        c = np.dot((ray.source - self.center), (ray.source - self.center)) - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / (2 * a)
            t2 = (-b - np.sqrt(delta)) / (2 * a)
            if (t1 < Epsilon and t1 > -Epsilon) or (t2 < Epsilon and t2 > -Epsilon):
                return None, None
            if t1 > 0 and t2 > 0:
                return self, min(t1, t2)
        return None, None


class Box(Shape):
    def __init__(self, center, edge_len: float):
        super().__init__()
        self.center = np.array(center)
        self.length = edge_len

    def intersect(self, ray: Ray):
        t_near = -float("inf")
        t_far = float("inf")
        co = self.center - ray.source
        for i in range(3):
            r = co[i]
            s = ray.direction[i]
            if abs(s) < Epsilon:
                t0 = r + self.length / 2
                if t0 > 0:
                    t0 = float("inf")
                else:
                    t0 = -float("inf")
                t1 = r - self.length / 2
                if t1 > 0:
                    t1 = float("inf")
                else:
                    t1 = -float("inf")
            else:
                t0 = (r + self.length / 2) / s
                t1 = (r - self.length / 2) / s
            if t0 > t1:
                tmp = t0
                t0 = t1
                t1 = tmp
            t_near = max(t_near, t0)
            t_far = min(t_far, t1)
            if t_near > t_far or t_far < 0:
                return None, None
        if t_near < 0:
            return None, None
        return self, t_near

    def get_normal(self, intersection_point):
        normal = [0, 0, 0]
        axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(3):
            if abs((intersection_point - self.center)[i] - self.length / 2) <= Epsilon:
                normal = axis[i]
                break
            if abs((intersection_point - self.center)[i] + self.length / 2) <= Epsilon:
                normal = -1 * axis[i]
                break
        return normal

    def normalized(self, intersection_point):
        return np.linalg.norm(intersection_point - self.center)
