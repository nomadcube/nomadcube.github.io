---
layout: page
title: All Posts
---

{% for category in site.tags %}
  <li>{{ category | first }}
    <ul>
    {% for posts in category %}
      {% for post in posts %}
        {% if post.title %}
          <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endif %}
      {% endfor %}
    {% endfor %}
    </ul>
  </li>
{% endfor %}
