from typing import Any

import numpy as np
from shapes_and_helper_methods import Ray, Box, Plane, Shape, Sphere, vector_normal, Light
import math
import sys
from PIL import Image

HEIGHT = 500
WIDTH = 500
SCENE_TXT = Any
IMAGE_PATH = Any
LIGHTS = Light
SHAPES = Shape
CAMERA = []
SET = []


def reshape(list_to_reshape, size):
    return np.array(list_to_reshape, dtype=float).reshape((int(len(list_to_reshape) / size), size))


def compute_image(position, look_at, screen_dist, screen_width):
    P0, background_color, image, right_vector, screen, up_vector = construct_screen(look_at,
                                                                                    position, screen_dist, screen_width)

    for i in range(int(HEIGHT)):
        P = np.copy(P0)
        for j, x in enumerate(np.linspace(screen[0], screen[1], int(WIDTH))):
            ray = shoot_ray_through_pixel(position, P)
            intersection_point, nearest_shape = find_nearest_intersection_for_ray(ray)
            if nearest_shape is None:
                output_color = background_color * np.array([255, 255, 255])
                image[-i, j] = output_color
            else:
                diffuse_and_specular = get_donation_from_lights(intersection_point, nearest_shape, position)
                current_color = (background_color * nearest_shape.transparency) + (
                        diffuse_and_specular * (1 - nearest_shape.transparency))
                tmp_normal = nearest_shape.get_normal(intersection_point)
                reflection = get_reflection(ray, tmp_normal, intersection_point, 1, SET[4], background_color)
                output_color = current_color + nearest_shape.reflection * reflection
                output_color = (np.clip(output_color, 0, 1) * np.array([255, 255, 255]))
                image[-i, j] = output_color
            P += right_vector
        P0 = P0 + up_vector
    return image


def construct_screen(look_at_point, position, screen_distance, screen_width):
    width = int(WIDTH)
    height = int(HEIGHT)
    screen_height = height * screen_width / width
    towards = np.array(look_at_point - position)
    towards_vector = towards / np.linalg.norm(towards)
    p_center = position + (screen_distance * towards_vector)
    background_color = SET[0:3]
    screen = (-(width / 2), width / 2, -(height / 2), height / 2)
    image = np.zeros((height, width, 3))
    if math.fabs(towards_vector[1]) == 1:
        up_vector, right_vector = calculate_vectors_aligned(towards_vector[1])
    else:
        up_vector, right_vector = calculate_vectors(
            build_matrix(towards_vector[0], towards_vector[1], towards_vector[2]))
    P0 = np.copy(p_center) - screen_width * 0.5 * right_vector - screen_height * 0.5 * up_vector
    right_vector = right_vector * (CAMERA[10] / width)
    up_vector = up_vector * (screen_width / height)
    return P0, background_color, image, right_vector, screen, up_vector


def shoot_ray_through_pixel(source, pixel):
    direction = pixel - source
    direction = vector_normal(direction)
    return Ray(source, direction)


def find_nearest_intersection_for_ray(ray):
    nearest_shape, minimum_distance = ray.nearest_intersection(SHAPES)
    if nearest_shape is None:
        return np.array((None, None, None)), None

    intersection_point = ray.source + (minimum_distance * vector_normal(ray.direction))

    return intersection_point, nearest_shape


def reflected_ray_direction(vector, normal):
    normal = vector_normal(normal)
    vector = vector_normal(vector)
    return vector - 2 * (np.dot(vector, normal)) * normal


def get_donation_from_lights(intersection_point, nearest_shape, position):
    sum_light = 0
    normal_at_intersection = nearest_shape.get_normal(intersection_point)
    for light in LIGHTS:
        sum_light += calculate_donation_per_light(normal_at_intersection, intersection_point, light, nearest_shape,
                                                  position)
    return sum_light


def calculate_donation_per_light(normal_at_intersection, intersection_point, light, nearest_shape, position):
    diffuse_arg = np.array(nearest_shape.diffuse)
    lgt = vector_normal(light.position - intersection_point)
    light_intensity = 1 - light.shadow_intensity + light.shadow_intensity * soft_shadows_calculation(lgt,
                                                                                                     light.position,
                                                                                                     light.radius,
                                                                                                     intersection_point)
    diffuse_donation = diffuse_arg * light.color * light_intensity * (np.dot(normal_at_intersection, lgt))
    specular_arg = np.array(nearest_shape.specular)
    V = vector_normal(intersection_point - position)
    R = vector_normal(reflected_ray_direction(lgt, normal_at_intersection))
    specular_donation = specular_arg * (light_intensity * (np.dot(R, V) ** nearest_shape.phong)) * light.color
    return diffuse_donation + specular_donation


def get_reflection(ray, normal, intersection_point, rec_depth, max_depth, background_color):
    if rec_depth < max_depth:
        rec_depth += 1
        reflected_ray = Ray(intersection_point, reflected_ray_direction(ray.direction, normal))
        new_intersection_reflected_point, reflection_nearest_shape = find_nearest_intersection_for_ray(reflected_ray)
        if reflection_nearest_shape is not None:
            curr_normal = reflection_nearest_shape.get_normal(new_intersection_reflected_point)
            diffuse_and_specular = get_donation_from_lights(new_intersection_reflected_point, reflection_nearest_shape,
                                                            intersection_point)
            return (background_color * reflection_nearest_shape.transparency) + (
                    (1 - reflection_nearest_shape.transparency) * diffuse_and_specular) + get_reflection(
                reflected_ray, curr_normal, new_intersection_reflected_point, rec_depth, max_depth,
                background_color) * reflection_nearest_shape.reflection
        return background_color
    return background_color


def soft_shadows_calculation(L, light_position, light_radius, intersection_point):
    shadow_rays_setting = SET[3]
    light_source_position, light_x, light_y = get_unit_vectors_for_rectangle(L, light_position,
                                                                             light_radius,
                                                                             shadow_rays_setting)
    hits = 0
    for i in range(int(shadow_rays_setting)):
        light_position_temp = light_source_position.copy()
        for j in range(int(shadow_rays_setting)):
            rand_v = light_position_temp + np.random.uniform(0, 1) * light_y
            new_v_direct = intersection_point - rand_v
            new_v_direct = vector_normal(new_v_direct)
            new_ray = Ray(rand_v, new_v_direct)
            tmp = new_ray.nearest_intersection(SHAPES)
            nearset_shape_as_arr = [tmp[1], tmp[0], tmp[0].get_normal(intersection_point)]
            if nearset_shape_as_arr[0] < float('inf') and np.allclose(rand_v + nearset_shape_as_arr[0] * new_v_direct,
                                                                      intersection_point, rtol=1e-02, atol=1e-02):
                hits += 1
            light_position_temp += light_y
        light_source_position += light_x
    return hits / math.pow(shadow_rays_setting, 2)


def get_unit_vectors_for_rectangle(lgt, light_position, light_radius, shadow_rays_setting):
    rand = np.random.randn(3)
    v1 = vector_normal(rand - np.dot(rand, lgt) * lgt)
    if not np.array_equiv(np.dot(v1, lgt), np.zeros((3, 0, 0))):
        raise Exception()
    v2 = np.cross(lgt, v1)
    light_source_position = light_position - 0.5 * light_radius * v1 - 0.5 * light_radius * v2
    if not np.allclose(light_source_position + 0.5 * light_radius * v1 + 0.5 * light_radius * v2, light_position):
        raise Exception()
    light_x = v1 * light_radius / shadow_rays_setting
    light_y = v2 * light_radius / shadow_rays_setting
    return light_source_position, light_x, light_y


def build_matrix(a, b, c):
    My = -b
    print(My)
    size = math.sqrt(1 - (My * My))
    print(size)
    Nx = -a / size
    print(Nx)
    Nz = c / size
    print(Nz)
    return np.array([[Nz, 0, Nx], [-My * Nx, size, My * Nz], [-size * Nx, -My, size * Nz]])


def calculate_vectors_aligned(b):
    if b == -1:
        return np.array([1, 0, 0]), np.array([0, 0, 1])
    if b == 1:
        return np.array([1, 0, 0]), np.array([0, 0, -1])


def calculate_vectors(matrix):
    up_vector = np.array([0, 1, 0])
    right_vector = np.array([1, 0, 0])
    return up_vector.dot(matrix), right_vector.dot(matrix)


def get_args():
    global HEIGHT, WIDTH, SCENE_TXT, IMAGE_PATH
    args = sys.argv[1:]
    img_h = 500
    img_w = 500
    scene_file = args[0]
    output_image_file = args[1]

    if len(args) > 2:
        img_w = args[2]
    if len(args) > 3:
        img_h = args[3]
    HEIGHT = img_h
    WIDTH = img_w
    SCENE_TXT = scene_file
    IMAGE_PATH = output_image_file


def build_lights(lights_list):
    lights = []
    for light in lights_list:
        our_lights = Light(light[0:3], light[3:6], light[6], light[7], light[8])
        lights.append(our_lights)
    return lights


def build_shapes(materials_list, planes_list, spheres_list, boxes_list):
    shapes = []
    build_shape(shapes, materials_list, spheres_list, "sphere")
    build_shape(shapes, materials_list, planes_list, "plane")
    build_shape(shapes, materials_list, boxes_list, "box")
    return shapes


def build_shape(shapes, materials_list, shape_list, type):
    for shape in shape_list:
        if type == 'sphere':
            new_shape = Sphere(shape[0:3], shape[3])
        elif type == 'box':
            new_shape = Box(shape[0:3], shape[3])
        else:
            new_shape = Plane(shape[0:3], shape[3])
        mat = materials_list[int(shape[4]) - 1]
        new_shape.material(mat[0:3], mat[3:6], mat[6:9], mat[9], mat[10])
        shapes.append(new_shape)


def main():
    get_args()
    read_txt()
    image = compute_image(CAMERA[0:3], CAMERA[3:6], CAMERA[9], CAMERA[10])
    im = Image.fromarray(image.astype('uint8'), 'RGB')
    im.save(IMAGE_PATH)


def read_txt():
    global CAMERA, SET, SHAPES, LIGHTS, WIDTH, HEIGHT, SCENE_TXT
    setting = []
    lights_list = []
    materials_list = []
    planes_list = []
    spheres_list = []
    boxes_list = []
    camera = []
    scene_file = open(SCENE_TXT, "r")
    lines = scene_file.readlines()
    for line in lines:
        if line[0] == '#' or line[0] == '\n':
            continue
        else:
            line = line.strip().split("\t")
            title = line[0]
            for word in line:
                word = word.strip().replace(" ", "").strip("\n")
                if word != '' and word != title and word != 'cam' and word != 'set':
                    word = float(word)
                    if title == 'cam ':
                        camera.append(word)
                    elif title == 'set ':
                        setting.append(word)
                    elif title == 'lgt':
                        lights_list.append(word)
                    elif title == 'mtl':
                        materials_list.append(word)
                    elif title == 'pln':
                        planes_list.append(word)
                    elif title == 'sph':
                        spheres_list.append(word)
                    elif title == 'box':
                        boxes_list.append(word)
    build_globals(boxes_list, camera, lights_list, materials_list, planes_list, setting, spheres_list)


def build_globals(boxes_list, camera, lights_list, materials_list, planes_list, setting, spheres_list):
    global CAMERA, SET, SHAPES, LIGHTS
    CAMERA = reshape(camera, 11)[0]
    SET = reshape(setting, 5)[0]
    materials_list = reshape(materials_list, 11)
    planes_list = reshape(planes_list, 5)
    spheres_list = reshape(spheres_list, 5)
    boxes_list = reshape(boxes_list, 5)
    lights_list = reshape(lights_list, 9)
    SHAPES = build_shapes(materials_list, planes_list, spheres_list, boxes_list)
    LIGHTS = build_lights(lights_list)


if __name__ == '__main__':
    main()
