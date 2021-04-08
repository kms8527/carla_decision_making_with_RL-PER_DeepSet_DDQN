import carla
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class KeyboardControl(object):
    def __init__(self, env, start_in_autopilot):
        pass
    def parse_events(self,client,env,clock):
        for event in pygame.event.get():
           if event.type == pygame.QUIT:
                return True
           elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    env.camera_rgb.toggle_camera()
                    # env.camera_semseg.toggle_camera()
                elif event.key == K_SPACE:
                    if self.agent.is_training is True:
                        self.agent.is_training = False
                    else:
                        self.agent.is_training = True

    @staticmethod
    def _is_quit_shortcut(key): #ctrl + q 또는 Escape 누를 시 종료
        return (key == K_ESCAPE) or (key == K_q) and (pygame.key.get_mods() & KMOD_CTRL)
