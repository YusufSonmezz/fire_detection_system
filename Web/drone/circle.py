from dronekit import connect, VehicleMode, LocationGlobalRelative, mavutil
import math
from time import sleep

class DroneController():
    def __init__(self, diameter:float, altitude:int, num_points:int):
        self.vehicle = connect('127.0.0.1:14550', wait_ready=True, timeout=60)
        self.home_location = self.vehicle.location.global_relative_frame

        self.diameter = diameter
        self.num_points = num_points
        self.altitude = altitude

    def get_coordinate(self, iter:int):
        radius = self.diameter / 2

        angle = 2.0 * math.pi * (iter / self.num_points)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)

        north = self.home_location.lat + x
        east = self.home_location.lon + y

        return LocationGlobalRelative(north, east, self.altitude)

    def go_to(self, iter: int):
        waypoint = self.get_coordinate(iter)
        self.vehicle.simple_goto(waypoint)

        while True:
            current_location = self.vehicle.location.global_relative_frame
            distance = self.distance_between_points(current_location, waypoint)
            
            if distance < 1:  # Adjust the threshold as needed
                break
            
            sleep(1)

    def distance_between_points(self, point1, point2):
        # Calculate the distance between two LocationGlobalRelative points
        dlat = point2.lat - point1.lat
        dlon = point2.lon - point1.lon
        return (dlat**2 + dlon**2)**0.5 * 1e5  # Multiply by a scaling factor for distance in meters

    
    def mode_guided(self):

        while not self.vehicle.is_armable:
            print("Drone is not urmable right now...")
            sleep(1)
        
        self.vehicle.mode = VehicleMode("GUIDED")
        while not self.vehicle.armed:
            print("Drone is armed.")
            self.vehicle.armed = True
            sleep(1)
    
    def mode_rtl(self):
        self.vehicle.mode = VehicleMode("RTL")
        sleep(1)
    
    def takeoff(self):

        self.vehicle.simple_takeoff(self.altitude)
        while True:
            if self.vehicle.location.global_relative_frame.alt >= self.altitude * 0.90:
                break
            sleep(1)
    
    def get_home_location(self):
        return self.home_location.lat, self.home_location.lon
    
    def get_current_location(self):
        return self.vehicle.location.global_relative_frame.lat, self.vehicle.location.global_relative_frame.lon, self.altitude
    
    def get_current_altitude(self):
        return self.vehicle.location.global_relative_frame.alt
    

if __name__ == "__main__":
    altitude = 20
    diameter = 0.001
    num_points = 36

    drone = DroneController(diameter, altitude, num_points)
    drone.mode_guided()
    drone.takeoff()
    
    for i in range(num_points):
        drone.go_to(i)
        drone.take_a_pic()
    
    
    drone.mode_rtl()
