# MIT License

# Copyright (c) 2020 Flavius Stefan Nicu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import glob
import itertools
import os
from random import randint, shuffle

import cv2
import numpy as np
from fpdf import FPDF
from rich import print
from rich.progress import track


COORDS_5_6 = [
    {"x": 375, "y": 40},
    {"x": 90, "y": 325},
    {"x": 700, "y": 325},
    {"x": 205, "y": 635},
    {"x": 570, "y": 635},
    {"x": 395, "y": 325},
]

COORDS_7 = [
    {"x": 205, "y": 40},
    {"x": 570, "y": 40},
    {"x": 90, "y": 325},
    {"x": 395, "y": 325},
    {"x": 700, "y": 325},
    {"x": 205, "y": 635},
    {"x": 570, "y": 635},
]

# Cards coords expresd in mm
PDF_CARDS_X = [8, 102, 8, 102, 8, 102]
PDF_CARDS_Y = [8, 8, 102, 102, 196, 196]

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

DOC_X = 1000
DOC_Y = DOC_X

CIRCLE_RADIUS = int(round(DOC_X / 2) - 10)
CIRCLE_CENTER = (int(DOC_Y / 2), int(DOC_X / 2))


def generateCards(numberOfSymb):
    """
    Source:	https://fr.wikipedia.org/wiki/Dobble
	"""
    nbSymByCard = numberOfSymb
    nbCards = (nbSymByCard ** 2) - nbSymByCard + 1
    cards = []
    n = nbSymByCard - 1
    t = []
    t.append([[(i + 1) + (j * n) for i in range(n)] for j in range(n)])
    for ti in range(n - 1):
        t.append(
            [
                [t[0][((ti + 1) * i) % n][(j + i) % n] for i in range(n)]
                for j in range(n)
            ]
        )
    t.append([[t[0][i][j] for i in range(n)] for j in range(n)])
    for i in range(n):
        t[0][i].append(nbCards - n)
        t[n][i].append(nbCards - n + 1)
        for ti in range(n - 1):
            t[ti + 1][i].append(nbCards - n + 1 + ti + 1)
    t.append([[(i + (nbCards - n)) for i in range(nbSymByCard)]])
    for ti in t:
        cards = cards + ti
    return cards


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image,
        rot_mat,
        image.shape[1::-1],
        flags=cv2.INTER_LINEAR,
        borderValue=WHITE_COLOR,
    )
    return result


def get_deck(args):
    cwd = os.getcwd()
    imgs_names = glob.glob(os.path.join(args.symbols_path, "*.png"))
    cards_list = generateCards(args.num_symbols)
    symb_nums = list(set(itertools.chain.from_iterable(cards_list)))

    print(
        f"{len(cards_list)} symbols would be need for {args.num_symbols} symbols cards."
    )
    print(f"{len(imgs_names)} images detected in the folder.")

    assert len(symb_nums) == len(imgs_names), print(
        f"[red]The number of needed symbols is different to the detected in the symbols folder. \n[bold]Please put {len(symb_nums)} symbols in the folder[/bold][/red]"
    )

    imgs_dicc = {}
    for num in symb_nums:
        imgs_dicc[num - 1] = imgs_names[num - 1]

    imgs_cards_list = []
    for card in cards_list:
        tmp = []
        for symb in card:
            tmp.append(imgs_dicc[symb - 1])
        imgs_cards_list.append(tmp)

    return imgs_cards_list


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def generate_dobble_PDF(args):

    results_path = os.path.join(args.out_path, "generated_cards")
    os.makedirs(results_path, exist_ok=True)

    deck = get_deck(args)

    if args.num_symbols != 7:
        cords_map = COORDS_5_6
    else:
        cords_map = COORDS_7

    pdf = FPDF()
    pdf.add_page()

    for card_n, card in track(
        enumerate(deck), total=len(deck), description="Generating cards...",
    ):
        card_image = np.ones((DOC_X, DOC_Y, 3), dtype=np.uint8)
        card_image[:] = WHITE_COLOR

        shuffle(card)
        for symb_n, symb_path in enumerate(card):
            s_img = cv2.imread(symb_path)
            s_img = rotate_bound(s_img, randint(0, 359))
            new_size = int(250 * randint(80, 120) / 100)
            s_img = cv2.resize(s_img, (new_size, new_size))

            y_offset = cords_map[symb_n]["y"]
            x_offset = cords_map[symb_n]["x"]
            card_image[
                y_offset : y_offset + s_img.shape[1],
                x_offset : x_offset + s_img.shape[0],
            ] = s_img
            repet = False

        cv2.circle(
            card_image,
            center=CIRCLE_CENTER,
            radius=CIRCLE_RADIUS,
            color=BLACK_COLOR,
            thickness=8,
        )

        out_img_path = os.path.join(results_path, f"card_{card_n}.png")
        cv2.imwrite(out_img_path, card_image)

        # If allready 6 cards on a PDF page
        if card_n % 6 == 0 and card_n != 0:
            pdf.add_page()
        pdf.image(
            out_img_path, PDF_CARDS_X[card_n % 6], PDF_CARDS_Y[card_n % 6], 85, 85
        )

    os.makedirs(args.out_path,exist_ok=True)
    out_path = os.path.join(args.out_path, args.out_name)
    pdf.output(out_path, "F")
    print(
        f"[bold green]Correctly generated!!! :clap: :smile: Open from {out_path}  [/bold green]"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_symbols",
        type=int,
        default=6,
        choices=[5, 6, 7],
        help="Nº of symbols to use in each card. Default:6 ",
    )
    parser.add_argument(
        "--symbols_path",
        type=str,
        default=os.path.join(os.getcwd(), "img"),
        help="Path where PNG symbols are located. Default: /img",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.getcwd(),
        help="Path where dobble PDF would be located. Default: /",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="dobble_to_print.pdf",
        help="Name of the output PDF file. Default: dobble_to_print",
    )

    args = parser.parse_args()
    generate_dobble_PDF(args)
