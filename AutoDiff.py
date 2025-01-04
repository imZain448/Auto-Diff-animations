from manim import *


class AutoDiffDemo(Scene):
    def construct(self):
        # Create nodes for a simple computation graph
        x = MathTex("x").shift(LEFT * 3)
        w = MathTex("w").shift(UP * 2)
        b = MathTex("b").shift(DOWN * 2)
        wx = MathTex("wx").shift(LEFT)
        wx_plus_b = MathTex("wx + b").shift(RIGHT)
        y = MathTex("y").shift(RIGHT * 3)

        # Add nodes to the scene
        self.play(Write(x), Write(w), Write(b))
        self.wait(1)

        # Create edges (arrows) between nodes
        self.play(TransformFromCopy(x, wx), TransformFromCopy(w, wx))
        self.play(
            TransformFromCopy(wx, wx_plus_b), TransformFromCopy(b, wx_plus_b)
        )
        self.play(TransformFromCopy(wx_plus_b, y))
        self.wait(1)

        # Add arrows to represent the computation flow
        self.play(
            Create(Arrow(x.get_right(), wx.get_left())),
            Create(Arrow(w.get_bottom(), wx.get_top())),
            Create(Arrow(wx.get_right(), wx_plus_b.get_left())),
            Create(Arrow(b.get_top(), wx_plus_b.get_bottom())),
            Create(Arrow(wx_plus_b.get_right(), y.get_left())),
        )
        self.wait(1)

        # Highlight the backpropagation process
        grad_y = MathTex("\\frac{\\partial L}{\\partial y}").next_to(y, UP)
        grad_wx_plus_b = MathTex(
            "\\frac{\\partial L}{\\partial (wx+b)}"
        ).next_to(wx_plus_b, UP)
        grad_wx = MathTex("\\frac{\\partial L}{\\partial wx}").next_to(wx, UP)
        grad_x = MathTex("\\frac{\\partial L}{\\partial x}").next_to(x, UP)
        grad_w = MathTex("\\frac{\\partial L}{\\partial w}").next_to(w, UP)
        grad_b = MathTex("\\frac{\\partial L}{\\partial b}").next_to(b, UP)

        self.play(Write(grad_y))
        self.play(TransformFromCopy(grad_y, grad_wx_plus_b))
        self.play(TransformFromCopy(grad_wx_plus_b, grad_wx))
        self.play(
            Transform(grad_wx, grad_x),
            Transform(grad_wx, grad_w),
        )
        self.play(TransformFromCopy(grad_wx_plus_b, grad_b))
        self.wait(2)


# ========================================
# ABOUT THE FUNCTION
# ----------------------------------------
# function to trace
# f(x1, x2) = x1x2 + exp(x1x2) - sin(x2)
# ----------------------------------------
# forward pass
# v1 = x1
# v2 = x2
# v3 = v1 * v2
# v4 = sin(v2)
# v5 = exp(v3)
# v6 = v3 - v4
# v7 = v6 + v5
# ----------------------------------------
# backward pass
# grad_v1 = 1
# grad_v2 = 0
# grad_v3 = v1*grad_v2 + v2*grad_v1
# grad_v4 = grad_v3 * exp(v3)
# grad_v5 = grad_v2 * cos(v2)
# grad_v6 = grad_v4 - grad_v5
# grad_v7 = grad_v6 + grad_v3
# ----------------------------------------


class AutoDiffForward(Scene):
    def construct(self):
        # Define the function in parts so it can be animated
        f = MathTex("f(").shift(LEFT * 4)
        x1 = MathTex("x_1").next_to(f, RIGHT)
        comma = MathTex(",").next_to(x1, RIGHT)
        x2 = MathTex("x_2").next_to(comma, RIGHT)
        bracket_end = MathTex(")").next_to(x2, RIGHT)
        equals = MathTex("=").next_to(bracket_end, RIGHT)
        x1x2 = MathTex("x_1x_2").next_to(equals, RIGHT)
        plus = MathTex("+").next_to(x1x2, RIGHT)
        exp_x1x2 = MathTex("e^{x_1x_2}").next_to(plus, RIGHT)
        minus = MathTex("-").next_to(exp_x1x2, RIGHT)
        sin_x2 = MathTex("sin(x_2)").next_to(minus, RIGHT)

        self.play(
            Write(f),
            Write(x1),
            Write(comma),
            Write(x2),
            Write(bracket_end),
            Write(equals),
        )
        self.play(
            Write(x1x2),
            Write(plus),
            Write(exp_x1x2),
            Write(minus),
            Write(sin_x2),
        )
        self.wait(1)

        # Fade out the unneeded parts
        self.play(FadeOut(f), FadeOut(comma), FadeOut(bracket_end))
        self.play(FadeOut(equals), FadeOut(minus), FadeOut(plus))
        self.wait(0.5)

        # Align and move each part using next_to
        # separating the intermediate nodes that will form the computation graph
        self.play(x1.animate.shift(RIGHT * 3 + UP * 2))
        self.play(x2.animate.next_to(x1, DOWN * 2, aligned_edge=LEFT))
        self.play(x1x2.animate.next_to(x2, DOWN * 2, aligned_edge=LEFT))
        self.play(sin_x2.animate.next_to(x1x2, DOWN * 2, aligned_edge=LEFT))
        self.play(
            exp_x1x2.animate.next_to(sin_x2, DOWN * 2, aligned_edge=LEFT)
        )
        self.wait(1)

        # add intermediate variables
        v1 = MathTex("v_1 = ").next_to(x1, LEFT)
        v2 = MathTex("v_2 = ").next_to(x2, LEFT)
        v3 = MathTex("v_3 = ").next_to(x1x2, LEFT)
        v4 = MathTex("v_4 = ").next_to(sin_x2, LEFT)
        v5 = MathTex("v_5 = ").next_to(exp_x1x2, LEFT)
        v6 = MathTex("v_6").next_to(v5, DOWN * 2, aligned_edge=LEFT)
        v7 = MathTex("v_7").next_to(v6, DOWN * 2, aligned_edge=LEFT)
        v6_val = MathTex("= v_3 - v_4").next_to(v6, aligned_edge=LEFT)
        v7_val = MathTex("= v_6 + v_5").next_to(v7, aligned_edge=LEFT)

        self.play(
            Write(v1),
            Write(v2),
            Write(v3),
            Write(v4),
            Write(v5),
            Write(v6),
            Write(v7),
            Write(v6_val),
            Write(v7_val),
        )
        self.wait(1)

        # keeping the full function defintion at the top

        func = (
            MathTex("f(x_1, x_2) = x_1x_2 + e^{x_1x_2} - sin(x_2)")
            .next_to(v1, LEFT * 1 + UP * 2, aligned_edge=LEFT)
            .shift(RIGHT)
        )

        self.play(Write(func), Write(SurroundingRectangle(func)))

        self.wait(1)

        # redefine v4 and v5 using intermediate variables
        exp_v3 = MathTex("e^{v_3}").next_to(v5)
        sin_v2 = MathTex("sin(v_2)").next_to(v4)

        # add information about the intermediate varaibles
        brace = Brace(VGroup(v1, v2, v3, v4, v5, v6, v7), LEFT)
        brace_lable = Text(
            """
            these are the 
            intermediate variables 

            they represent 
            the nodes of 
            the computation graphs.
            
            and the operators 
            will form the edges 
            of the computation graphs.
            """,
            font_size=18,
            # color=YELLOW,
            # background_stroke_width=2,
            # background_stroke_color=YELLOW,
            # background=True,
            # background_fill_color=YELLOW,
            # background_fill_opacity=0.5,
        ).next_to(brace, LEFT, buff=0.2)

        self.play(Write(brace), Write(brace_lable))

        # Add annotations with arrows pointing to the rewritten expressions
        annotation1 = Text(
            "Rewriting to intermediate variable", font_size=18, color=YELLOW
        ).next_to(exp_v3, RIGHT * 8 + UP * 0.5)
        arrow1 = Arrow(annotation1.get_left(), exp_v3.get_right())
        arrow2 = Arrow(annotation1.get_left(), sin_v2.get_right())
        annotation_surround = SurroundingRectangle(annotation1, buff=0.2)

        # play all the animations
        self.play(
            FadeOut(exp_x1x2),
            Write(exp_v3),
            Write(annotation1),
            Write(annotation_surround),
            # Write(SurroundingRectangle(annotation1)),
            Write(arrow1),
            Write(arrow2),
        )
        self.wait(0.5)
        self.play(
            FadeOut(sin_x2),
            Write(sin_v2),
        )
        self.wait(1)
        self.play(
            FadeOut(arrow1),
            FadeOut(arrow2),
            FadeOut(annotation_surround),
            FadeOut(annotation1),
            FadeOut(brace),
            FadeOut(brace_lable),
        )
        self.wait(0.5)

        # addiing the surrounding circles to the vs
        v1_circle = Circle(radius=0.4, color=GREEN).move_to(v1.get_center())
        v2_circle = Circle(radius=0.4, color=GREEN).move_to(v2.get_center())
        v3_circle = Circle(radius=0.4, color=GREEN).move_to(v3.get_center())
        v4_circle = Circle(radius=0.4, color=GREEN).move_to(v4.get_center())
        v5_circle = Circle(radius=0.4, color=GREEN).move_to(v5.get_center())
        v6_circle = Circle(radius=0.4, color=GREEN).move_to(v6.get_center())
        v7_circle = Circle(radius=0.4, color=GREEN).move_to(v7.get_center())

        v1_new = MathTex("v_1").move_to(v1.get_center())
        v2_new = MathTex("v_2").move_to(v2.get_center())
        v3_new = MathTex("v_3").move_to(v3.get_center())
        v4_new = MathTex("v_4").move_to(v4.get_center())
        v5_new = MathTex("v_5").move_to(v5.get_center())

        self.play(
            FadeOut(x1),
            FadeOut(x2),
            FadeOut(x1x2),
            FadeOut(exp_v3),
            FadeOut(sin_v2),
            FadeOut(v6_val),
            FadeOut(v7_val),
            ReplacementTransform(v1, v1_new),
            ReplacementTransform(v2, v2_new),
            ReplacementTransform(v3, v3_new),
            ReplacementTransform(v4, v4_new),
            ReplacementTransform(v5, v5_new),
            Write(v1_circle),
            Write(v2_circle),
            Write(v3_circle),
            Write(v4_circle),
            Write(v5_circle),
            Write(v6_circle),
            Write(v7_circle),
        )

        vc1 = VGroup(v1_new, v1_circle)
        vc2 = VGroup(v2_new, v2_circle)
        vc3 = VGroup(v3_new, v3_circle)
        vc4 = VGroup(v4_new, v4_circle)
        vc5 = VGroup(v5_new, v5_circle)
        vc6 = VGroup(v6, v6_circle)
        vc7 = VGroup(v7, v7_circle)

        self.wait(1)
        self.play(
            vc1.animate.shift(LEFT * 2 + DOWN),
        )
        self.play(
            vc2.animate.next_to(vc1, DOWN * 3),
        )
        self.play(
            vc3.animate.next_to(vc1, RIGHT * 5),
        )
        self.play(
            vc4.animate.next_to(vc2, RIGHT * 5),
        )
        self.play(
            vc5.animate.next_to(vc3, RIGHT * 5),
        )
        self.play(
            vc6.animate.next_to(vc4, RIGHT * 5),
        )
        self.play(
            vc7.animate.next_to(vc5, RIGHT * 5),
        )

        arrow_v1 = Arrow(vc1.get_left() + LEFT * 2, vc1.get_left())
        arrow_v2 = Arrow(vc2.get_left() + LEFT * 2, vc2.get_left())
        arrow_v1v3 = Arrow(vc1.get_right(), vc3.get_left())
        arrow_v2v3 = Arrow(vc2.get_right(), vc3.get_left())
        arrow_v2v4 = Arrow(vc2.get_right(), vc4.get_left())
        arrow_v3v5 = Arrow(vc3.get_right(), vc5.get_left())
        arrow_v4v6 = Arrow(vc4.get_right(), vc6.get_left())
        arrow_v3v6 = Arrow(vc3.get_right(), vc6.get_left())
        arrow_v5v7 = Arrow(vc5.get_right(), vc7.get_left())
        arrow_v6v7 = Arrow(vc6.get_right(), vc7.get_left())
        arrow_v7 = Arrow(vc7.get_right(), vc7.get_right() + RIGHT * 2)
        self.play(
            Create(arrow_v1),
            Create(arrow_v2),
            Create(arrow_v1v3),
            Create(arrow_v2v3),
            Create(arrow_v2v4),
            Create(arrow_v3v5),
            Create(arrow_v4v6),
            Create(arrow_v3v6),
            Create(arrow_v5v7),
            Create(arrow_v6v7),
            Create(arrow_v7),
        )

        # Add descriptor on each node.
        v1_descriptor = MathTex("x_1", font_size=24, color=YELLOW).next_to(
            vc1, UP
        )
        v2_descriptor = MathTex("x_2", font_size=24, color=YELLOW).next_to(
            vc2, DOWN
        )
        v3_descriptor = MathTex("x_1x_2", font_size=24, color=YELLOW).next_to(
            vc3, UP
        )
        v4_descriptor = MathTex(
            "sin(v_2)", font_size=24, color=YELLOW
        ).next_to(vc4, DOWN)
        v5_descriptor = MathTex("e^{v_3}", font_size=24, color=YELLOW).next_to(
            vc5, UP
        )
        v6_descriptor = MathTex(
            "v_3 - v_4", font_size=24, color=YELLOW
        ).next_to(vc6, DOWN)
        v7_descriptor = MathTex(
            "v_6 + v_5", font_size=24, color=YELLOW
        ).next_to(vc7, UP)

        self.play(
            Write(v1_descriptor),
            Write(v2_descriptor),
            Write(v3_descriptor),
            Write(v4_descriptor),
            Write(v5_descriptor),
            Write(v6_descriptor),
            Write(v7_descriptor),
        )

        forward_pass = Text(
            "Forward Pass", font_size=46, color=YELLOW
        ).next_to(vc6, DOWN * 4 + LEFT * 2)
        forward_pass_arrow = Arrow(
            forward_pass.get_left() + DOWN * 0.5,
            forward_pass.get_right() + DOWN * 0.5,
        )
        self.play(Write(forward_pass), Create(forward_pass_arrow))
        self.wait(1)
        self.play(FadeOut(forward_pass), FadeOut(forward_pass_arrow))
        self.wait(2)


class AutoDiffBackward(Scene):
    def construct(self):
        # Create nodes for backward pass
        grad_v7 = MathTex("\\frac{\\partial L}{\\partial v_7}").shift(
            RIGHT * 4
        )
        grad_v6 = MathTex("\\frac{\\partial L}{\\partial v_6}").shift(
            RIGHT * 2
        )
        grad_v4 = MathTex("\\frac{\\partial L}{\\partial v_4}").shift(UP * 2)
        grad_v5 = MathTex("\\frac{\\partial L}{\\partial v_5}").shift(DOWN * 2)
        grad_v3 = MathTex("\\frac{\\partial L}{\\partial v_3}").shift(LEFT * 2)
        grad_v1 = MathTex("\\frac{\\partial L}{\\partial v_1}").shift(
            LEFT * 4 + UP * 2
        )
        grad_v2 = MathTex("\\frac{\\partial L}{\\partial v_2}").shift(
            LEFT * 4 + DOWN * 2
        )

        # Add nodes
        self.play(Write(grad_v7))
        self.wait(0.5)
        self.play(Write(grad_v6))
        self.play(Write(grad_v4), Write(grad_v5))
        self.play(Write(grad_v3))
        self.play(Write(grad_v1), Write(grad_v2))

        # Add arrows with gradient formulas
        arrows = [
            Arrow(grad_v7.get_left(), grad_v6.get_right()),
            Arrow(grad_v7.get_left(), grad_v3.get_right()),
            Arrow(grad_v6.get_left(), grad_v4.get_right()),
            Arrow(grad_v6.get_left(), grad_v5.get_right()),
            Arrow(grad_v3.get_left(), grad_v1.get_right()),
            Arrow(grad_v3.get_left(), grad_v2.get_right()),
        ]

        formulas = [
            MathTex("\\cdot 1").next_to(arrows[0], UP, buff=0.2).scale(0.7),
            MathTex("\\cdot 1").next_to(arrows[1], DOWN, buff=0.2).scale(0.7),
            MathTex("\\cdot e^{v_3}")
            .next_to(arrows[2], UP, buff=0.2)
            .scale(0.7),
            MathTex("\\cdot -1").next_to(arrows[3], DOWN, buff=0.2).scale(0.7),
            MathTex("\\cdot v_2").next_to(arrows[4], UP, buff=0.2).scale(0.7),
            MathTex("\\cdot v_1")
            .next_to(arrows[5], DOWN, buff=0.2)
            .scale(0.7),
        ]

        for arrow, formula in zip(arrows, formulas):
            self.play(Create(arrow), Write(formula), run_time=0.5)

        self.wait(2)
