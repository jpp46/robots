import math
import numpy as np

HEIGHT = 0.3
EPS = 0.05

def make_robot(sim, weight_matrix):
    main_body = sim.send_box(x=0, y=0, z=HEIGHT+EPS,
                             length=HEIGHT, width=HEIGHT,
                             height=EPS*2.0, mass=1)
    light_sensor = sim.send_light_sensor(body_id=main_body)

    # id arrays
    thighs = [0]*4
    shins = [0]*4
    hips = [0]*4
    knees = [0]*4
    foot_sensors = [0]*4
    sensor_neurons = [0]*5
    motor_neurons = [0]*8

    delta = float(math.pi)/2.0

    # quadruped is a box with one leg on each side
    # each leg consists thigh and shin cylinders
    # with hip and knee joints
    # each shin/foot then has a touch sensor
    for i in range(4):
        theta = delta*i
        x_pos = math.cos(theta)*HEIGHT
        y_pos = math.sin(theta)*HEIGHT

        thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                      r1=x_pos, r2=y_pos, r3=0,
                                      length=HEIGHT, radius=EPS, capped=True
                                      )

        hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                       x=x_pos/2.0, y=y_pos/2.0, z=HEIGHT+EPS,
                                       n1=-y_pos, n2=x_pos, n3=0,
                                       lo=-math.pi/4.0, hi=math.pi/4.0,
                                       speed=1.0)

        motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

        x_pos2 = math.cos(theta)*1.5*HEIGHT
        y_pos2 = math.sin(theta)*1.5*HEIGHT

        shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                     r1=0, r2=0, r3=1,
                                     length=HEIGHT, radius=EPS,
                                     mass=1., capped=True)

        knees[i] = sim.send_hinge_joint(thighs[i], shins[i],
                                        x=x_pos2, y=y_pos2, z=HEIGHT+EPS,
                                        n1=-y_pos, n2=x_pos, n3=0,
                                        lo=-math.pi/4.0, hi=math.pi/4.0)

        motor_neurons[i+4] = sim.send_motor_neuron(knees[i])
        foot_sensors[i] = sim.send_touch_sensor(shins[i])
        sensor_neurons[i] = sim.send_sensor_neuron(foot_sensors[i])

    sensor_neurons[-1] = sim.send_sensor_neuron(light_sensor)
    for (source_id, i) in zip(sensor_neurons, range(5)):
        for (target_id, j) in zip(motor_neurons, range(8)):
            sim.send_synapse(source_id, target_id, weight_matrix[i, j])

    env_box = sim.send_box(x=0, y=HEIGHT*30, z=HEIGHT/2.0,
                           length=HEIGHT, width=HEIGHT, height=HEIGHT)
    sim.send_light_source(env_box)

    return light_sensor

def run(sim):
  sim.start()

def fitness(sim, light_sensor):
  sim.wait_to_finish()
  return sim.get_sensor_data(light_sensor)[-1]
  

    