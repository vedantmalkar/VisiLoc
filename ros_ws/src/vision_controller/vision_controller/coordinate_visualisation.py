#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tkinter as tk
import threading


class PositionVisualizer(Node):
    
    def __init__(self):
        super().__init__("position_visualizer")
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        self.position_sub = self.create_subscription(
            PoseStamped,
            "/robot_position",
            self.position_callback,
            10
        )

        self.canvas = None
        self.window = None
        self.robot_pos = None

        self.tk_thread = threading.Thread(target=self.init_tkinter, daemon=True)    #create thread to allow simultaneous running
        self.tk_thread.start()
        
        self.get_logger().info("Position Visualizer Started!")
    
    def position_callback(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.get_logger().info(f"bot position: X={self.robot_x:.2f}, Y={self.robot_y:.2f}")
    
    def init_tkinter(self):
        self.window = tk.Tk()
        self.window.title("Robot Position Visualiser")
        
        self.canvas = tk.Canvas(self.window, width=1000, height=1000, bg='white')
        self.canvas.pack()
        
        for i in range(0, 1001, 100):
            self.canvas.create_line(i, 0, i, 1000, fill='lightgray')
            self.canvas.create_line(0, i, 1000, i, fill='lightgray')

        self.canvas.create_line(500, 0, 500, 1000, fill='black', width=2)
        self.canvas.create_line(0, 500, 1000, 500, fill='black', width=2)

        self.canvas.create_oval(495, 495, 505, 505, fill='black')
        self.canvas.create_text(520, 480, text="(0,0)", font=('Arial', 10, 'bold'))

        self.canvas.create_text(500, 20, text="+X", font=('Arial', 12, 'bold'))
        self.canvas.create_text(500, 980, text="-X", font=('Arial', 12, 'bold'))
        self.canvas.create_text(20, 500, text="+Y", font=('Arial', 12, 'bold'))
        self.canvas.create_text(980, 500, text="-Y", font=('Arial', 12, 'bold'))
        
        self.window.after(200, self.update_canvas)
        self.window.mainloop()
    
    def world_to_canvas_x(self, world_x):
        return int(500 - (world_x * 100))
    
    def world_to_canvas_y(self, world_y):
        return int(500 - (world_y * 100))
    
    def update_canvas(self):
        if self.canvas is not None:
            canvas_x = self.world_to_canvas_y(self.robot_y)
            canvas_y = self.world_to_canvas_x(self.robot_x)

            if self.robot_pos:
                self.canvas.delete(self.robot_pos)
                self.canvas.delete("robot_label")

            self.robot_pos = self.canvas.create_oval(
                canvas_x - 8, canvas_y - 8,
                canvas_x + 8, canvas_y + 8,
                fill='blue', outline='darkblue', width=2
            )
        
        if self.window is not None:
            self.window.after(100, self.update_canvas)


def main(args=None):
    rclpy.init(args=args)
    node = PositionVisualizer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()