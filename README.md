<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/puchee99/JOBcn-DS-2022">
    <img src="images/pytorch.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">JOBcn-DS-2022</h3>

  <p align="center">
    Online Data Science hackathon  (JOBarcelona 2022)
    <br />
    <a href="https://github.com/puchee99/JOBcn-DS-2022"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/puchee99/JOBcn-DS-2022">View Demo</a>
    ·
    <a href="https://github.com/puchee99/JOBcn-DS-2022/issues">Report Bug</a>
    ·
    <a href="https://github.com/puchee99/JOBcn-DS-2022/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

The objective of this project is to classify insects according to the value of different sensors.
We have the data in a `.csv` where the `Insect` column is the target to predict.


Distribution of the data in the variables with a greater correlation with `y`:

![product-screenshot]

[Features image][product-screenshot]


<p align="right">(<a href="#top">back to top</a>)</p>


### Built With

* [Pytorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Logging](https://docs.python.org/3/library/logging.html)
* [Seaborn](https://seaborn.pydata.org/)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Installation


First, clone the repository:
   ```sh
   git clone https://github.com/puchee99/JOBcn-DS-2022.git
   ```
Access to the project folder with:
  ```sh
  cd JOBcn-DS-2022
  ```

We will create a virtual environment with `python3`
* Create environment with python 3 
    ```sh
    python3 -m venv venv
    ```
    
* Enable the virtual environment
    ```sh
    source venv/bin/activate
    ```

* Install the python dependencies on the virtual environment
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage
The train.py and test.py documents can be executed with bash using different arguments.

* To get the information of the arguments use:
    ```sh
    python name_document.py -h
    ```
    Example:
    ```sh
    python train.py -h
    ```
* To train the models use:
    ```sh
    python train.py
    ```
* To test the models use:
    ```sh
    python test.py
    ```


## Results

In the best training we have an accuracy of 0.923 predicting test set(20% of data from `train.csv`).

The predictions from the `test_x.csv` file can be found in the `output/results` folder in the `results.csv` file.

#### ACCURACY:
![accuracy]
[accuracy image][accuracy]

#### CONFUSION MATRIX:
![CM]
[CM image][CM]

#### LOSS:
![loss]
[loss image][loss]

#### ROC:
![roc]
[roc image][roc]

<!-- CONTACT -->
## Contact

Arnau Puche  - [@arnau_puche_vila](https://www.linkedin.com/in/arnau-puche-vila-ds/) - arnaupuchevila@gmail.com

Project Link: [https://github.com/puchee99/JOBcn-DS-2022](https://github.com/puchee99/JOBcn-DS)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/puchee99/JOBcn-DS-2022.svg?style=for-the-badge
[contributors-url]: https://github.com/puchee99/JOBcn-DS-2022/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/puchee99/JOBcn-DS-2022.svg?style=for-the-badge
[forks-url]: https://github.com/puchee99/JOBcn-DS-2022/network/members
[stars-shield]: https://img.shields.io/github/stars/puchee99/JOBcn-DS-2022.svg?style=for-the-badge
[stars-url]: https://github.com/puchee99/JOBcn-DS-2022/stargazers
[issues-shield]: https://img.shields.io/github/issues/puchee99/JOBcn-DS-2022.svg?style=for-the-badge
[issues-url]: https://github.com/puchee99/JOBcn-DS-2022/issues
[license-shield]: https://img.shields.io/github/license/puchee99/JOBcn-DS-2022.svg?style=for-the-badge
[license-url]: https://github.com/puchee99/JOBcn-DS-2022/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/arnau-puche-vila-ds/
[product-screenshot]: output/plots/features_distribution.png
[accuracy]: output/plots/MulticlassSimpleClassification_accuracy.png
[CM]: output/plots/MulticlassSimpleClassification_cm.png
[loss]: output/plots/MulticlassSimpleClassification_loss.png
[roc]: output/plots/MulticlassSimpleClassification_roc.png