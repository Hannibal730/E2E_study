"""Draw a click-through arrow inside the simulator's private Xephyr display."""
import time

from Xlib import X, display
from Xlib.ext import shape


def main():
    connection = display.Display()
    screen = connection.screen()
    root = screen.root
    width, height = 22, 28
    arrow = [(1, 1), (1, 23), (7, 17), (12, 26), (16, 24), (11, 15), (20, 15)]
    mask = root.create_pixmap(width, height, 1)
    gc = mask.create_gc(foreground=0)
    mask.fill_rectangle(gc, 0, 0, width, height)
    gc.change(foreground=1)
    mask.fill_poly(gc, X.Complex, X.CoordModeOrigin, arrow)

    image = root.create_pixmap(width, height, screen.root_depth)
    ink = image.create_gc(foreground=screen.white_pixel)
    image.fill_rectangle(ink, 0, 0, width, height)
    ink.change(foreground=screen.black_pixel, line_width=2)
    image.poly_line(ink, X.CoordModeOrigin, arrow + arrow[:1])
    window = root.create_window(0, 0, width, height, 0, X.CopyFromParent,
                                background_pixmap=image, override_redirect=True)
    window.set_wm_name("Udacity cursor")
    window.shape_mask(shape.SO.Set, shape.SK.Bounding, 0, 0, mask)
    window.shape_rectangles(shape.SO.Set, shape.SK.Input, X.Unsorted, 0, 0, [])
    window.map()
    while True:
        pointer = root.query_pointer()
        window.configure(x=pointer.root_x - 1, y=pointer.root_y - 1, stack_mode=X.Above)
        connection.flush()
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
