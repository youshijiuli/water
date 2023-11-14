import pygame
import sys
import random

class SnakeGame:
   def __init__(self):
       pygame.init()
       self.screen = pygame.display.set_mode((800, 600))
       pygame.display.set_caption("贪吃蛇")
       self.snake = Snake()
       self.food = Food()
       self.clock = pygame.time.Clock()
       self.score = 0

   def run(self):
       while True:
           self.screen.fill((255, 255, 255))
           self.snake.move()
           self.check_collision()
           self.draw()
           self.clock.tick(10)
           pygame.display.flip()

   def check_collision(self):
       if self.snake.head_pos == self.food.pos:
           self.score += 1
           self.snake.grow()
           self.food = Food()

   def draw(self):
       self.snake.draw()
       self.food.draw()
       pygame.display.set_caption(f"贪吃蛇 - 得分: {self.score}")

class Snake:
   def __init__(self):
       self.positions = [(100, 100), (120, 100), (140, 100)]
       self.direction = (20, 0)

   def move(self):
       new_position = (self.positions[0][0] + self.direction[0], self.positions[0][1] + self.direction[1])
       self.positions.insert(0, new_position)
       self.positions.pop()

   def grow(self):
       self.positions.append(self.positions[-1])

   def draw(self):
       for position in self.positions:
           pygame.draw.rect(game.screen, (0, 255, 0), pygame.Rect(position[0], position[1], 20, 20))

   def check_collision(self):
       # 检查蛇头是否碰到边界或自身
       if self.positions[0][0] < 0 or self.positions[0][0] >= 800 or self.positions[0][1] < 0 or self.positions[0][1] >= 600:
           game.run()
       for i in range(1, len(self.positions)):
           if self.positions[0] == self.positions[i]:
               game.run()

class Food:
   def __init__(self):
       self.pos = (random.randint(0, 39) * 20, random.randint(0, 29) * 20)

   def draw(self):
       pygame.draw.rect(game.screen, (255, 0, 0), pygame.Rect(self.pos[0], self.pos[1], 20, 20))
       
       
       
       
       
def main():
   global game
   game = SnakeGame()
   game.run()

if __name__ == "__main__":
   main()