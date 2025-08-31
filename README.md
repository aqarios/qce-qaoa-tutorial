# Constraint-Driven QAOA Design

> A tutorial by [Aqarios GmbH](https:://www.aqarios.com)

[https://aqarios-my.sharepoint.com/:p:/p/david_bucher/EZPjH2_c6I1JsrLGk6xl9BwBt0BZthReACltFgPBhP8xsQ?e=8muCXe](Link to Tutorial Slides)

This tutorial is composed of six notebooks that will teach you
- [01_QAOA](tutorials/01_QAOA.ipynb): The basics of the Quantum Approximate Optimization
  Algorithm (QAOA)
- [02_PenaltyTerms](tutorials/02_PenaltyTerms.ipynb): Reformulating constriants into
objectives in an automatic manner.
- [03_XYMixers](tutorials/03_XYMixers.ipynb): Automatic Handling of One-Hot Constraints
  via XY-Mixers
- [04_Aqarios_Luna](tutorials/04_Aqarios_Luna.ipynb): Using the Aqarios LunaSolve
platform for solving your optimization problems. See how _FlexQAOA_ does constraint
handling automatically.
- [05_Benchmarking](tutorials/05_Benchmarking.ipynb): Best Practices in Benchmarking
Quantum Optimization Algorithms
- [06_ModelExtension](tutorials/06_ModelExtension.ipynb): Getting to know more
capabilities of FlexQAOA.

We hope you have fun exploring this tutorial! 🚀

## Python Setup

Use a python version `>=3.11`.

We recommend using `uv` for dependency management and installation ([astral/uv](https://docs.astral.sh/uv/)). Simply run
```
uv sync
```
and every dependency will be installed.

Otherwise, you can use your preferred dependency management, e.g. pip
```
pip install -r requirements.txt
pip install -e .
```

## Luna Registration

To register to Luna, please visit [app.aqarios.com](http://app.aqarios.com).
Plese complete the steps you are ask and log into the platform.

You should be able to see the **Luna API Key** field on the top right. Copy the key and
put it into the `.env` file.
```
echo "LUNA_API_KEY=<THE_API_KEY>" > .env
```

---

## 💡 Ready to dive deeper?

Explore more tutorials, documentation, and resources to accelerate your journey

<img src="https://docs.aqarios.com/assets/aqarios.png#only-light" width="400px" alt="Aqarios Logo" />

[![Website](https://img.shields.io/badge/🌐_Website-Visit_Aqarios.com-blue?style=for-the-badge)](https://www.aqarios.com)
[![Documentation](https://img.shields.io/badge/📚_Documentation-Explore_Docs-green?style=for-the-badge)](https://docs.aqarios.com)
[![LinkedIn](https://img.shields.io/badge/🤝_LinkedIn-Connect_With_Us-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/aqarios-gmbh/)


**What's Next?**

- **Explore our documentation** for advanced features and best practices
- **Join our community** on LinkedIn for updates and discussions  
- **Check out more tutorials** to expand your skills

### 💬 Need Help?

Have questions or feedback about this tutorial? We'd love to hear from you! Connect with us through any of the links above.

---

<div align="center">
<small>

Tutorial provided by Aqarios GmbH | © 2025 Aqarios GmbH | Made with ❤️ for developers

</small>
</div>
